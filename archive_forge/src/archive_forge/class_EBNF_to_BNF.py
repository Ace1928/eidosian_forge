import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
import pkgutil
from ast import literal_eval
from contextlib import suppress
from typing import List, Tuple, Union, Callable, Dict, Optional, Sequence, Generator
from .utils import bfs, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique, small_factors, OrderedSet
from .lexer import Token, TerminalDef, PatternStr, PatternRE, Pattern
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol, TOKEN_DEFAULT_PRIORITY
from .utils import classify, dedup_list
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError, UnexpectedInput
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
@inline_args
class EBNF_to_BNF(Transformer_InPlace):

    def __init__(self):
        self.new_rules = []
        self.rules_cache = {}
        self.prefix = 'anon'
        self.i = 0
        self.rule_options = None

    def _name_rule(self, inner: str):
        new_name = '__%s_%s_%d' % (self.prefix, inner, self.i)
        self.i += 1
        return new_name

    def _add_rule(self, key, name, expansions):
        t = NonTerminal(name)
        self.new_rules.append((name, expansions, self.rule_options))
        self.rules_cache[key] = t
        return t

    def _add_recurse_rule(self, type_: str, expr: Tree):
        try:
            return self.rules_cache[expr]
        except KeyError:
            new_name = self._name_rule(type_)
            t = NonTerminal(new_name)
            tree = ST('expansions', [ST('expansion', [expr]), ST('expansion', [t, expr])])
            return self._add_rule(expr, new_name, tree)

    def _add_repeat_rule(self, a, b, target, atom):
        """Generate a rule that repeats target ``a`` times, and repeats atom ``b`` times.

        When called recursively (into target), it repeats atom for x(n) times, where:
            x(0) = 1
            x(n) = a(n) * x(n-1) + b

        Example rule when a=3, b=4:

            new_rule: target target target atom atom atom atom

        """
        key = (a, b, target, atom)
        try:
            return self.rules_cache[key]
        except KeyError:
            new_name = self._name_rule('repeat_a%d_b%d' % (a, b))
            tree = ST('expansions', [ST('expansion', [target] * a + [atom] * b)])
            return self._add_rule(key, new_name, tree)

    def _add_repeat_opt_rule(self, a, b, target, target_opt, atom):
        """Creates a rule that matches atom 0 to (a*n+b)-1 times.

        When target matches n times atom, and target_opt 0 to n-1 times target_opt,

        First we generate target * i followed by target_opt, for i from 0 to a-1
        These match 0 to n*a - 1 times atom

        Then we generate target * a followed by atom * i, for i from 0 to b-1
        These match n*a to n*a + b-1 times atom

        The created rule will not have any shift/reduce conflicts so that it can be used with lalr

        Example rule when a=3, b=4:

            new_rule: target_opt
                    | target target_opt
                    | target target target_opt

                    | target target target
                    | target target target atom
                    | target target target atom atom
                    | target target target atom atom atom

        """
        key = (a, b, target, atom, 'opt')
        try:
            return self.rules_cache[key]
        except KeyError:
            new_name = self._name_rule('repeat_a%d_b%d_opt' % (a, b))
            tree = ST('expansions', [ST('expansion', [target] * i + [target_opt]) for i in range(a)] + [ST('expansion', [target] * a + [atom] * i) for i in range(b)])
            return self._add_rule(key, new_name, tree)

    def _generate_repeats(self, rule: Tree, mn: int, mx: int):
        """Generates a rule tree that repeats ``rule`` exactly between ``mn`` to ``mx`` times.
        """
        if mx < REPEAT_BREAK_THRESHOLD:
            return ST('expansions', [ST('expansion', [rule] * n) for n in range(mn, mx + 1)])
        mn_target = rule
        for a, b in small_factors(mn, SMALL_FACTOR_THRESHOLD):
            mn_target = self._add_repeat_rule(a, b, mn_target, rule)
        if mx == mn:
            return mn_target
        diff = mx - mn + 1
        diff_factors = small_factors(diff, SMALL_FACTOR_THRESHOLD)
        diff_target = rule
        diff_opt_target = ST('expansion', [])
        for a, b in diff_factors[:-1]:
            diff_opt_target = self._add_repeat_opt_rule(a, b, diff_target, diff_opt_target, rule)
            diff_target = self._add_repeat_rule(a, b, diff_target, rule)
        a, b = diff_factors[-1]
        diff_opt_target = self._add_repeat_opt_rule(a, b, diff_target, diff_opt_target, rule)
        return ST('expansions', [ST('expansion', [mn_target] + [diff_opt_target])])

    def expr(self, rule: Tree, op: Token, *args):
        if op.value == '?':
            empty = ST('expansion', [])
            return ST('expansions', [rule, empty])
        elif op.value == '+':
            return self._add_recurse_rule('plus', rule)
        elif op.value == '*':
            new_name = self._add_recurse_rule('star', rule)
            return ST('expansions', [new_name, ST('expansion', [])])
        elif op.value == '~':
            if len(args) == 1:
                mn = mx = int(args[0])
            else:
                mn, mx = map(int, args)
                if mx < mn or mn < 0:
                    raise GrammarError("Bad Range for %s (%d..%d isn't allowed)" % (rule, mn, mx))
            return self._generate_repeats(rule, mn, mx)
        assert False, op

    def maybe(self, rule: Tree):
        keep_all_tokens = self.rule_options and self.rule_options.keep_all_tokens
        rule_size = FindRuleSize(keep_all_tokens).transform(rule)
        empty = ST('expansion', [_EMPTY] * rule_size)
        return ST('expansions', [rule, empty])