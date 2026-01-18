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
def will_not_get_removed(sym):
    if isinstance(sym, NonTerminal):
        return not sym.name.startswith('_')
    if isinstance(sym, Terminal):
        return keep_all_tokens or not sym.filter_out
    assert False