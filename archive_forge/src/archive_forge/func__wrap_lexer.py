from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
def _wrap_lexer(lexer_class):
    future_interface = getattr(lexer_class, '__future_interface__', False)
    if future_interface:
        return lexer_class
    else:

        class CustomLexerWrapper(Lexer):

            def __init__(self, lexer_conf):
                self.lexer = lexer_class(lexer_conf)

            def lex(self, lexer_state, parser_state):
                return self.lexer.lex(lexer_state.text)
        return CustomLexerWrapper