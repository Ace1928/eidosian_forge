import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
from io import open
import pkgutil
from ast import literal_eval
from numbers import Integral
from .utils import bfs, Py36, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique
from .lexer import Token, TerminalDef, PatternStr, PatternRE
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol
from .utils import classify, suppress, dedup_list, Str
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
def _mangle_exp(exp, mangle):
    if mangle is None:
        return exp
    exp = deepcopy(exp)
    for t in exp.iter_subtrees():
        for i, c in enumerate(t.children):
            if isinstance(c, Token) and c.type in ('RULE', 'TERMINAL'):
                t.children[i] = Token(c.type, mangle(c.value))
    return exp