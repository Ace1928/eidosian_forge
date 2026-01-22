from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .parsers.grammar_analysis import GrammarAnalyzer
from .lexer import LexerThread, TraditionalLexer, ContextualLexer, Lexer, Token, TerminalDef
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf
import re
class MakeParsingFrontend:

    def __init__(self, parser_type, lexer_type):
        self.parser_type = parser_type
        self.lexer_type = lexer_type

    def __call__(self, lexer_conf, parser_conf, options):
        assert isinstance(lexer_conf, LexerConf)
        assert isinstance(parser_conf, ParserConf)
        parser_conf.parser_type = self.parser_type
        lexer_conf.lexer_type = self.lexer_type
        return ParsingFrontend(lexer_conf, parser_conf, options)

    @classmethod
    def deserialize(cls, data, memo, lexer_conf, callbacks, options):
        parser_conf = ParserConf.deserialize(data['parser_conf'], memo)
        parser = LALR_Parser.deserialize(data['parser'], memo, callbacks, options.debug)
        parser_conf.callbacks = callbacks
        return ParsingFrontend(lexer_conf, parser_conf, options, parser=parser)