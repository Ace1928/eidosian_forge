import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
class SentinelNodeFinder(object):
    result = None

    def __init__(self, frame, stmts, tree, lasti, source):
        assert_(stmts)
        self.frame = frame
        self.tree = tree
        self.code = code = frame.f_code
        self.is_pytest = is_rewritten_by_pytest(code)
        if self.is_pytest:
            self.ignore_linenos = frozenset(assert_linenos(tree))
        else:
            self.ignore_linenos = frozenset()
        self.decorator = None
        self.instruction = instruction = self.get_actual_current_instruction(lasti)
        op_name = instruction.opname
        extra_filter = lambda e: True
        ctx = type(None)
        typ = type(None)
        if op_name.startswith('CALL_'):
            typ = ast.Call
        elif op_name.startswith(('BINARY_SUBSCR', 'SLICE+')):
            typ = ast.Subscript
            ctx = ast.Load
        elif op_name.startswith('BINARY_'):
            typ = ast.BinOp
            op_type = dict(BINARY_POWER=ast.Pow, BINARY_MULTIPLY=ast.Mult, BINARY_MATRIX_MULTIPLY=getattr(ast, 'MatMult', ()), BINARY_FLOOR_DIVIDE=ast.FloorDiv, BINARY_TRUE_DIVIDE=ast.Div, BINARY_MODULO=ast.Mod, BINARY_ADD=ast.Add, BINARY_SUBTRACT=ast.Sub, BINARY_LSHIFT=ast.LShift, BINARY_RSHIFT=ast.RShift, BINARY_AND=ast.BitAnd, BINARY_XOR=ast.BitXor, BINARY_OR=ast.BitOr)[op_name]
            extra_filter = lambda e: isinstance(e.op, op_type)
        elif op_name.startswith('UNARY_'):
            typ = ast.UnaryOp
            op_type = dict(UNARY_POSITIVE=ast.UAdd, UNARY_NEGATIVE=ast.USub, UNARY_NOT=ast.Not, UNARY_INVERT=ast.Invert)[op_name]
            extra_filter = lambda e: isinstance(e.op, op_type)
        elif op_name in ('LOAD_ATTR', 'LOAD_METHOD', 'LOOKUP_METHOD'):
            typ = ast.Attribute
            ctx = ast.Load
            extra_filter = lambda e: attr_names_match(e.attr, instruction.argval)
        elif op_name in ('LOAD_NAME', 'LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_DEREF', 'LOAD_CLASSDEREF'):
            typ = ast.Name
            ctx = ast.Load
            extra_filter = lambda e: e.id == instruction.argval
        elif op_name in ('COMPARE_OP', 'IS_OP', 'CONTAINS_OP'):
            typ = ast.Compare
            extra_filter = lambda e: len(e.ops) == 1
        elif op_name.startswith(('STORE_SLICE', 'STORE_SUBSCR')):
            ctx = ast.Store
            typ = ast.Subscript
        elif op_name.startswith('STORE_ATTR'):
            ctx = ast.Store
            typ = ast.Attribute
            extra_filter = lambda e: attr_names_match(e.attr, instruction.argval)
        else:
            raise RuntimeError(op_name)
        with lock:
            exprs = {cast(EnhancedAST, node) for stmt in stmts for node in ast.walk(stmt) if isinstance(node, typ) if isinstance(getattr(node, 'ctx', None), ctx) if extra_filter(node) if statement_containing_node(node) == stmt}
            if ctx == ast.Store:
                self.result = only(exprs)
                return
            matching = list(self.matching_nodes(exprs))
            if not matching and typ == ast.Call:
                self.find_decorator(stmts)
            else:
                self.result = only(matching)

    def find_decorator(self, stmts):
        stmt = only(stmts)
        assert_(isinstance(stmt, (ast.ClassDef, function_node_types)))
        decorators = stmt.decorator_list
        assert_(decorators)
        line_instructions = [inst for inst in self.clean_instructions(self.code) if inst.lineno == self.frame.f_lineno]
        last_decorator_instruction_index = [i for i, inst in enumerate(line_instructions) if inst.opname == 'CALL_FUNCTION'][-1]
        assert_(line_instructions[last_decorator_instruction_index + 1].opname.startswith('STORE_'))
        decorator_instructions = line_instructions[last_decorator_instruction_index - len(decorators) + 1:last_decorator_instruction_index + 1]
        assert_({inst.opname for inst in decorator_instructions} == {'CALL_FUNCTION'})
        decorator_index = decorator_instructions.index(self.instruction)
        decorator = decorators[::-1][decorator_index]
        self.decorator = decorator
        self.result = stmt

    def clean_instructions(self, code):
        return [inst for inst in get_instructions(code) if inst.opname not in ('EXTENDED_ARG', 'NOP') if inst.lineno not in self.ignore_linenos]

    def get_original_clean_instructions(self):
        result = self.clean_instructions(self.code)
        if not any((inst.opname == 'JUMP_IF_NOT_DEBUG' for inst in self.compile_instructions())):
            result = [inst for inst in result if inst.opname != 'JUMP_IF_NOT_DEBUG']
        return result

    def matching_nodes(self, exprs):
        original_instructions = self.get_original_clean_instructions()
        original_index = only((i for i, inst in enumerate(original_instructions) if inst == self.instruction))
        for expr_index, expr in enumerate(exprs):
            setter = get_setter(expr)
            assert setter is not None
            replacement = ast.BinOp(left=expr, op=ast.Pow(), right=ast.Str(s=sentinel))
            ast.fix_missing_locations(replacement)
            setter(replacement)
            try:
                instructions = self.compile_instructions()
            finally:
                setter(expr)
            if sys.version_info >= (3, 10):
                try:
                    handle_jumps(instructions, original_instructions)
                except Exception:
                    if TESTING or expr_index < len(exprs) - 1:
                        continue
                    raise
            indices = [i for i, instruction in enumerate(instructions) if instruction.argval == sentinel]
            for index_num, sentinel_index in enumerate(indices):
                sentinel_index -= index_num * 2
                assert_(instructions.pop(sentinel_index).opname == 'LOAD_CONST')
                assert_(instructions.pop(sentinel_index).opname == 'BINARY_POWER')
            for index_num, sentinel_index in enumerate(indices):
                sentinel_index -= index_num * 2
                new_index = sentinel_index - 1
                if new_index != original_index:
                    continue
                original_inst = original_instructions[original_index]
                new_inst = instructions[new_index]
                if original_inst.opname == new_inst.opname in ('CONTAINS_OP', 'IS_OP') and original_inst.arg != new_inst.arg and (original_instructions[original_index + 1].opname != instructions[new_index + 1].opname == 'UNARY_NOT'):
                    instructions.pop(new_index + 1)
                if sys.version_info < (3, 10):
                    for inst1, inst2 in zip_longest(original_instructions, instructions):
                        assert_(inst1 and inst2 and opnames_match(inst1, inst2))
                yield expr

    def compile_instructions(self):
        module_code = compile_similar_to(self.tree, self.code)
        code = only(self.find_codes(module_code))
        return self.clean_instructions(code)

    def find_codes(self, root_code):
        checks = [attrgetter('co_firstlineno'), attrgetter('co_freevars'), attrgetter('co_cellvars'), lambda c: is_ipython_cell_code_name(c.co_name) or c.co_name]
        if not self.is_pytest:
            checks += [attrgetter('co_names'), attrgetter('co_varnames')]

        def matches(c):
            return all((f(c) == f(self.code) for f in checks))
        code_options = []
        if matches(root_code):
            code_options.append(root_code)

        def finder(code):
            for const in code.co_consts:
                if not inspect.iscode(const):
                    continue
                if matches(const):
                    code_options.append(const)
                finder(const)
        finder(root_code)
        return code_options

    def get_actual_current_instruction(self, lasti):
        """
        Get the instruction corresponding to the current
        frame offset, skipping EXTENDED_ARG instructions
        """
        instructions = list(get_instructions(self.code))
        index = only((i for i, inst in enumerate(instructions) if inst.offset == lasti))
        while True:
            instruction = instructions[index]
            if instruction.opname != 'EXTENDED_ARG':
                return instruction
            index += 1