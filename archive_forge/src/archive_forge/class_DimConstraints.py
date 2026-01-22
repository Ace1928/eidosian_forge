import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
class DimConstraints:
    """
    Custom solver for a system of constraints on symbolic dimensions.
    Solutions are "static" values or simplified "dynamic" constraints.
    """

    def __init__(self, symbol_to_source, var_to_val, marked_dynamic, source_name_to_debug_name):
        self._univariate_inequalities: Dict[sympy.Symbol, Set[sympy.Expr]] = defaultdict(set)
        self._symbols_with_equalities: Set[sympy.Symbol] = set()
        self._substitutions: Dict[sympy.Symbol, sympy.Integer] = {}
        self._var_to_val: Dict[sympy.Symbol, sympy.Integer] = var_to_val
        self._congruences: Set[sympy.Expr] = defaultdict(set)
        self._multivariate_inequalities: Set[sympy.Expr] = set()
        self._symbolic_equivalences: List[Tuple[Source, sympy.Expr]] = []
        self._static_results: Set[str] = set()
        self._dynamic_results: Set[str] = set()
        self._dcp = DynamicDimConstraintPrinter(symbol_to_source, source_name_to_debug_name)
        self._inconsistencies: List[str] = []
        self._marked_dynamic = marked_dynamic

    def rewrite_with_congruences(self, s, expr):
        """
        Eliminate expressions of the form b // d and b % d while adding congruences of the form b % d == k.
        This leaves rational operators (in particular of the form b / d) that our inequality solver can handle.
        We solve the added congruences separately (using our congruence solver, see below).
        """

        def mod_handler(*args):
            base, divisor = args
            base, divisor = (self.rewrite_with_congruences(s, base), self.rewrite_with_congruences(s, divisor))
            mod_reduced = base.subs(self._var_to_val) % divisor.subs(self._var_to_val)
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return mod_reduced

        def floor_div_handler(*args):
            base, divisor = args
            base, divisor = (self.rewrite_with_congruences(s, base), self.rewrite_with_congruences(s, divisor))
            mod_reduced = base.subs(self._var_to_val) % divisor.subs(self._var_to_val)
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return (base - mod_reduced) / divisor
        if expr.has(Mod):
            expr = expr.replace(Mod, mod_handler)
        if expr.has(FloorDiv):
            expr = expr.replace(FloorDiv, floor_div_handler)
        return expr

    def add(self, expr) -> bool:
        if expr == sympy.true:
            return True
        orig_expr = expr
        orig_reduced = orig_expr.subs(self._var_to_val)
        if orig_reduced == sympy.false:
            self._inconsistencies.append(f'{orig_expr} is inconsistent!')
        free_symbols = expr.free_symbols
        assert free_symbols, f'Did not expect constraint with no free variables: {expr}'
        if len(free_symbols) > 1:
            self._multivariate_inequalities.add(expr)
        else:
            s = next(iter(free_symbols))
            expr = self.rewrite_with_congruences(s, expr)
            if expr == sympy.true:
                return True
            reduced = expr.subs(self._var_to_val)
            if reduced == sympy.false:
                self._inconsistencies.append(f'{expr}, obtained by rewriting {orig_expr} with congruences, is inconsistent!')
            if isinstance(expr, sympy.Eq):
                self._symbols_with_equalities.add(s)
            self._univariate_inequalities[s].add(expr)
        return False

    def add_equality(self, source, expr):
        if expr.is_number:
            self._static_results.add(f'{source.name()} == {expr}')
        else:
            self._symbolic_equivalences.append((source, expr))

    def reduce_congruences(self):
        reduced_congruences = {}
        for s, congruences in self._congruences.items():
            remainder_modulus_pairs = []
            congruences_to_check = set()
            for congruence in congruences:
                base, divisor = congruence.args
                tmp = sympy.Symbol('tmp', integer=True)
                symbol, solution = sympy.solve_linear(base - divisor * tmp, symbols=[s])
                if s == symbol:
                    modulus, remainder = sympy.polys.polytools.div(solution, tmp)
                    if isinstance(modulus, sympy.Integer) and isinstance(remainder, sympy.Integer):
                        remainder = remainder % modulus
                        remainder_modulus_pairs.append((remainder, modulus))
                        continue
                congruences_to_check.add(congruence)
            if remainder_modulus_pairs:
                remainder, modulus = sympy.ntheory.modular.solve_congruence(*remainder_modulus_pairs)
                reduced_congruences[s] = {(s - remainder) % modulus}
                substitution = {s: modulus * sympy.Symbol('tmp', integer=True) + remainder}
                reduced_congruences[s].update((congruence for congruence in congruences_to_check if not sympy.checksol(congruence, substitution)))
            else:
                reduced_congruences[s] = congruences_to_check
        return reduced_congruences

    def raise_inconsistencies(self):
        if self._inconsistencies:
            msg = '\n'.join(self._inconsistencies)
            self._inconsistencies.clear()
            raise ValueError(f'The following inconsistencies were found:\n{msg}')

    def _force_specialization(self, s):
        val = self._var_to_val[s]
        self._static_results.add(f'{self._dcp.symbol_to_source[s][0].name()} == {val}')
        self._substitutions[s] = val

    def specialize_divisor_symbols(self):
        for expr in self._multivariate_inequalities:
            for atom in expr.atoms(FloorDiv, Mod):
                _, divisor = atom.args
                for s in divisor.free_symbols:
                    self._force_specialization(s)
        multivariate_inequalities = self._multivariate_inequalities
        self._multivariate_inequalities = set()
        for expr in multivariate_inequalities:
            self.add(expr.subs(self._substitutions))
        self.raise_inconsistencies()
        self._univariate_inequalities = {s: exprs for s, exprs in self._univariate_inequalities.items() if s not in self._substitutions}
        self._congruences = {s: congruences for s, congruences in self._congruences.items() if s not in self._substitutions}

    def solve(self, disable_congruences=True, disable_equivalences=True):
        self.raise_inconsistencies()
        while self._symbols_with_equalities:
            s = self._symbols_with_equalities.pop()
            exprs = self._univariate_inequalities.pop(s)
            solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
            if isinstance(solution, sympy.And):
                solution = next((arg for arg in solution.args if isinstance(arg, sympy.Eq)), solution)
            assert isinstance(solution, sympy.Eq), f'Expected an equality constraint for {s}, got {solution}'
            symbol, val = solution.args
            assert symbol == s, f'Expected a constraint on {s} instead of on {symbol}'
            self._static_results.add(f'{self._dcp.symbol_to_source[s][0].name()} == {val}')
            self._substitutions[s] = val
            multivariate_inequalities = self._multivariate_inequalities
            self._multivariate_inequalities = set()
            for expr in multivariate_inequalities:
                self.add(expr.subs(s, self._substitutions[s]))
            self.raise_inconsistencies()
        self.specialize_divisor_symbols()
        reduced_congruences = self.reduce_congruences()
        for s, congruences in reduced_congruences.items():
            for congruence in congruences:
                if s not in self._substitutions or not sympy.checksol(congruence, {s: self._substitutions[s]}):
                    if disable_congruences:
                        self._force_specialization(s)
                        self._univariate_inequalities.pop(s, None)
                    else:
                        self._dynamic_results.add(self._dcp.doprint(sympy.Eq(congruence, 0)))
        for s, exprs in self._univariate_inequalities.items():
            try:
                solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
                if isinstance(solution, sympy.And):
                    for arg in solution.args:
                        self._dynamic_results.add(self._dcp.doprint(arg))
                else:
                    self._dynamic_results.add(self._dcp.doprint(solution))
            except NotImplementedError as e:
                log.warning('Failed to reduce inequalities: %s', e)
                for expr in exprs:
                    self._dynamic_results.add(self._dcp.doprint(expr))
        symbolic_equivalences = self._symbolic_equivalences
        self._symbolic_equivalences = []
        for source, expr in symbolic_equivalences:
            if disable_equivalences and (not isinstance(expr, sympy.Symbol)):
                for s in expr.free_symbols:
                    self._force_specialization(s)
                    sexpr = self._dcp._print_Symbol(s)
                    self._dynamic_results = {r for r in self._dynamic_results if sexpr not in r}
            self.add_equality(source, expr.subs(self._substitutions))
        for source, expr in self._symbolic_equivalences:
            self._dynamic_results.add(f'{self._dcp.print_source(source)} == {self._dcp.doprint(expr)}')

    def forced_specializations(self):

        def debug_name(src):
            name = src.name()
            if self._dcp.source_name_to_debug_name:
                return f'{self._dcp.source_name_to_debug_name[name]} = {name}'
            else:
                return name
        return {debug_name(self._dcp.symbol_to_source[s][0]): val for s, val in self._substitutions.items() if s in self._marked_dynamic}

    def remove_redundant_dynamic_results(self):
        candidates_for_removal = []
        dynamic_results = set()
        for dc in self._dynamic_results:
            dc_ = re.sub('2 <= dynamic_dim(.+)', 'dynamic_dim\\1', dc)
            if dc != dc_:
                candidates_for_removal.append(dc_)
            else:
                dynamic_results.add(dc_)
        for dc in candidates_for_removal:
            found = False
            for other_dc in dynamic_results:
                if dc in other_dc:
                    found = True
            if not found:
                dynamic_results.add(dc)
        self._dynamic_results = dynamic_results

    def prettify_results(self, original_signature: inspect.Signature, constraint_violation_error=None, forced_specializations=None):
        if self._dcp.source_name_to_debug_name:

            def transform(s):
                for k, v in self._dcp.source_name_to_debug_name.items():
                    s = s.replace(k, v)
                return s
            results = defaultdict(dict)

            def flip(op):
                if op == '<=':
                    return '>='
                if op == '>=':
                    return '<='
                if op == '<':
                    return '>'
                if op == '>':
                    return '<'
                assert op == '=='
                return op

            def relation_with_digit(expr, op, digit):
                if op == '<=':
                    results[expr]['max'] = digit
                elif op == '<':
                    results[expr]['max'] = digit - 1
                elif op == '>=':
                    results[expr]['min'] = digit
                elif op == '>':
                    results[expr]['min'] = digit + 1
                else:
                    assert op == '=='
                    results[expr]['eq'] = digit
            for s in self._static_results.union(self._dynamic_results):
                t = transform(s)
                if t == s:
                    continue
                left, op, right = t.split(' ')
                if op == '==' and left == right:
                    continue
                if right.isdigit():
                    relation_with_digit(left, op, int(right))
                elif left.isdigit():
                    relation_with_digit(right, flip(op), int(left))
                else:
                    assert op == '=='
                    results[left]['eq'] = right
            buf = ''
            debug_names = set()
            if forced_specializations:
                debug_names.update((k.split(' = ')[0] for k in forced_specializations.keys()))
                buf += f'Specializations unexpectedly required ({', '.join(debug_names)})! For more information, run with TORCH_LOGS=dynamic.\n'
                for s, val in forced_specializations.items():
                    buf += f'  - {s} must be specialized to {val} because the guards generated for it are too complex.\n'
            dims = []
            others = []
            match = None
            if constraint_violation_error:
                match = re.search('Constraints violated \\((.*)\\)', constraint_violation_error.args[0])
            if match is not None:
                debug_names.update(match.expand('\\1').split(', '))
            for k, c in results.items():
                if k not in debug_names:
                    continue
                if 'eq' in c:
                    other = c['eq']
                    if isinstance(other, int):
                        others.append(f'{k} = None  # {other}')
                    else:
                        others.append(f'{k} = {other}')
                else:
                    min_ = c.get('min', None)
                    if min_ == 2:
                        min_ = None
                    max_ = c.get('max', None)
                    if min_ is not None and max_ is not None:
                        dims.append(f"{k} = Dim('{k}', min={min_}, max={max_})")
                    elif min_ is not None:
                        dims.append(f"{k} = Dim('{k}', min={min_})")
                    elif max_ is not None:
                        dims.append(f"{k} = Dim('{k}', max={max_})")
                    else:
                        dims.append(f"{k} = Dim('{k}')")
            buf += '\nSuggested fixes:\n  '
            buf += '\n  '.join(dims + others)
            return buf

        def extract_and_rewrite_local(dc):
            match = re.search("L\\['(.+?)'\\]", dc)
            if match is None:
                return
            arg = match.expand('\\1')
            dc = re.sub("L\\['(.+?)'\\]", '\\1', dc)
            return (arg, dc)

        def group(results, args_index):
            groups = defaultdict(list)
            for dc in results:
                local = extract_and_rewrite_local(dc)
                if local is None:
                    continue
                arg, dc = local
                if arg in args_index:
                    groups[args_index[arg]].append(dc)
                else:
                    continue
            sorted_groups = []
            for idx, dcs in sorted(groups.items()):
                _, arg = idx
                sorted_groups.append((arg, sorted(dcs)))
            return sorted_groups
        signature = original_signature.replace(return_annotation=inspect.Signature.empty)
        args_index = {}
        for i, arg in enumerate(signature.parameters.keys()):
            args_index[arg] = (i, arg)

        def print_results(grouped, indent, result_fn):
            nonlocal buf
            space = False
            for arg, results in grouped:
                if space:
                    buf += '\n'
                else:
                    space = True
                buf += f'\n{indent}# {arg}:'
                for result in results:
                    buf += f'\n{indent}{result_fn(result)}'
        buf = ''
        if forced_specializations:
            buf += 'Some dynamic dimensions need to be specialized because the constraints inferred for them are too complex to specify.\n'
            for s, val in forced_specializations.items():
                buf += f'  - {s}, which was marked dynamic, must be specialized to {val}.\n'
        indent = 4 * ' '
        if self._static_results:
            grouped_static_results = group(self._static_results, args_index)
            buf += '\nThe following dimensions have been specialized and CANNOT be dynamic.'
            buf += f'\n```\ndef specializations{str(signature)}:'
            print_results(grouped_static_results, indent, lambda result: f'assert {result}')
            buf += '\n```\n'
        if self._dynamic_results:
            grouped_dynamic_results = group(self._dynamic_results, args_index)
            buf += '\nThe following dimensions CAN be dynamic.'
            buf += '\nPlease use the following code to specify the constraints they must satisfy:'
            buf += f'\n```\ndef specify_constraints{str(signature)}:'
            buf += f'\n{indent}return ['
            print_results(grouped_dynamic_results, indent * 2, lambda result: f'{result},')
            buf += f'\n{indent}]\n```\n'
        return buf