from abc import ABC, abstractmethod
import getpass
import sys, os, pickle
import tempfile
import types
import re
from typing import (
from .exceptions import ConfigurationError, assert_config, UnexpectedInput
from .utils import Serialize, SerializeMemoizer, FS, isascii, logger
from .load_grammar import load_grammar, FromPackageLoader, Grammar, verify_used_files, PackageResource, sha256_digest
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
from .lexer import Lexer, BasicLexer, TerminalDef, LexerThread, Token
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import _validate_frontend_args, _get_lexer_callbacks, _deserialize_parsing_frontend, _construct_parsing_frontend
from .grammar import Rule
def _prepare_callbacks(self) -> None:
    self._callbacks = {}
    if self.options.ambiguity != 'forest':
        self._parse_tree_builder = ParseTreeBuilder(self.rules, self.options.tree_class or Tree, self.options.propagate_positions, self.options.parser != 'lalr' and self.options.ambiguity == 'explicit', self.options.maybe_placeholders)
        self._callbacks = self._parse_tree_builder.create_callback(self.options.transformer)
    self._callbacks.update(_get_lexer_callbacks(self.options.transformer, self.terminals))