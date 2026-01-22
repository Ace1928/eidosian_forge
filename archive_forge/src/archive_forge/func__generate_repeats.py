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