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