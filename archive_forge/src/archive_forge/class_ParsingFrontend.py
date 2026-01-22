from typing import Any, Callable, Dict, Optional, Collection, Union, TYPE_CHECKING
from .exceptions import ConfigurationError, GrammarError, assert_config
from .utils import get_regexp_width, Serialize
from .lexer import LexerThread, BasicLexer, ContextualLexer, Lexer
from .parsers import earley, xearley, cyk
from .parsers.lalr_parser import LALR_Parser
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
class ParsingFrontend(Serialize):
    __serialize_fields__ = ('lexer_conf', 'parser_conf', 'parser')
    lexer_conf: LexerConf
    parser_conf: ParserConf
    options: Any

    def __init__(self, lexer_conf: LexerConf, parser_conf: ParserConf, options, parser=None):
        self.parser_conf = parser_conf
        self.lexer_conf = lexer_conf
        self.options = options
        if parser:
            self.parser = parser
        else:
            create_parser = _parser_creators.get(parser_conf.parser_type)
            assert create_parser is not None, '{} is not supported in standalone mode'.format(parser_conf.parser_type)
            self.parser = create_parser(lexer_conf, parser_conf, options)
        lexer_type = lexer_conf.lexer_type
        self.skip_lexer = False
        if lexer_type in ('dynamic', 'dynamic_complete'):
            assert lexer_conf.postlex is None
            self.skip_lexer = True
            return
        if isinstance(lexer_type, type):
            assert issubclass(lexer_type, Lexer)
            self.lexer = _wrap_lexer(lexer_type)(lexer_conf)
        elif isinstance(lexer_type, str):
            create_lexer = {'basic': create_basic_lexer, 'contextual': create_contextual_lexer}[lexer_type]
            self.lexer = create_lexer(lexer_conf, self.parser, lexer_conf.postlex, options)
        else:
            raise TypeError('Bad value for lexer_type: {lexer_type}')
        if lexer_conf.postlex:
            self.lexer = PostLexConnector(self.lexer, lexer_conf.postlex)

    def _verify_start(self, start=None):
        if start is None:
            start_decls = self.parser_conf.start
            if len(start_decls) > 1:
                raise ConfigurationError('Lark initialized with more than 1 possible start rule. Must specify which start rule to parse', start_decls)
            start, = start_decls
        elif start not in self.parser_conf.start:
            raise ConfigurationError('Unknown start rule %s. Must be one of %r' % (start, self.parser_conf.start))
        return start

    def _make_lexer_thread(self, text: str) -> Union[str, LexerThread]:
        cls = self.options and self.options._plugins.get('LexerThread') or LexerThread
        return text if self.skip_lexer else cls.from_text(self.lexer, text)

    def parse(self, text: str, start=None, on_error=None):
        chosen_start = self._verify_start(start)
        kw = {} if on_error is None else {'on_error': on_error}
        stream = self._make_lexer_thread(text)
        return self.parser.parse(stream, chosen_start, **kw)

    def parse_interactive(self, text: Optional[str]=None, start=None):
        chosen_start = self._verify_start(start)
        if self.parser_conf.parser_type != 'lalr':
            raise ConfigurationError("parse_interactive() currently only works with parser='lalr' ")
        stream = self._make_lexer_thread(text)
        return self.parser.parse_interactive(stream, chosen_start)