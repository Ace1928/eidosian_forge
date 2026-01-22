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
class ApplyTemplates(Transformer_InPlace):
    """Apply the templates, creating new rules that represent the used templates"""

    def __init__(self, rule_defs):
        self.rule_defs = rule_defs
        self.replacer = _ReplaceSymbols()
        self.created_templates = set()

    def template_usage(self, c):
        name = c[0].name
        args = c[1:]
        result_name = '%s{%s}' % (name, ','.join((a.name for a in args)))
        if result_name not in self.created_templates:
            self.created_templates.add(result_name)
            (_n, params, tree, options), = (t for t in self.rule_defs if t[0] == name)
            assert len(params) == len(args), args
            result_tree = deepcopy(tree)
            self.replacer.names = dict(zip(params, args))
            self.replacer.transform(result_tree)
            self.rule_defs.append((result_name, [], result_tree, deepcopy(options)))
        return NonTerminal(result_name)