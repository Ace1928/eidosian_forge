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
def _unpack_definition(self, tree, mangle):
    if tree.data == 'rule':
        name, params, exp, opts = _make_rule_tuple(*tree.children)
        is_term = False
    else:
        name = tree.children[0].value
        params = ()
        opts = int(tree.children[1]) if len(tree.children) == 3 else TOKEN_DEFAULT_PRIORITY
        exp = tree.children[-1]
        is_term = True
    if mangle is not None:
        params = tuple((mangle(p) for p in params))
        name = mangle(name)
    exp = _mangle_definition_tree(exp, mangle)
    return (name, is_term, exp, params, opts)