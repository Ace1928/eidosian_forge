from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
def _verify_start(self, start=None):
    if start is None:
        start_decls = self.parser_conf.start
        if len(start_decls) > 1:
            raise ConfigurationError('Lark initialized with more than 1 possible start rule. Must specify which start rule to parse', start_decls)
        start, = start_decls
    elif start not in self.parser_conf.start:
        raise ConfigurationError('Unknown start rule %s. Must be one of %r' % (start, self.parser_conf.start))
    return start