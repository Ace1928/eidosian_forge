from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterator, Optional, Set, Tuple, Union
import interegular
from interegular.fsm import FSM
from interegular.patterns import Unsupported
from lark import Lark, Token
from lark.common import LexerConf, ParserConf
from lark.exceptions import LexError, UnexpectedInput
from lark.indenter import Indenter
from lark.lexer import (
from lark.parser_frontends import (
from lark.parsers.lalr_analysis import (
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.parsers.lalr_parser import LALR_Parser, ParseConf, ParserState, _Parser
from outlines.fsm.regex import (
class PartialLark(Lark):
    __serialize_fields__ = ('parser', 'rules', 'options', 'deterministic', 'use_value_stack')

    def __init__(self, grammar, **options):
        self.deterministic = options.pop('deterministic', False)
        self.use_value_stack = options.pop('use_value_stack', False)
        options['regex'] = True
        super().__init__(grammar, **options)
        assert self.options.parser == 'lalr'

    def _build_lexer(self, dont_ignore: bool=False) -> 'PartialBasicLexer':
        lexer_conf = self.lexer_conf
        if dont_ignore:
            from copy import copy
            lexer_conf = copy(lexer_conf)
            lexer_conf.ignore = ()
        return PartialBasicLexer(lexer_conf)

    def _build_parser(self) -> 'PartialParsingFrontend':
        self._prepare_callbacks()
        _validate_frontend_args(self.options.parser, self.options.lexer)
        parser_conf = PartialParserConf(self.rules, self._callbacks, self.options.start, self.deterministic, self.use_value_stack)
        parser_type = self.options.parser
        lexer_type = self.options.lexer
        lexer_conf = self.lexer_conf
        assert isinstance(lexer_conf, LexerConf)
        assert isinstance(parser_conf, ParserConf)
        parser_conf.parser_type = parser_type
        self.lexer_conf.lexer_type = lexer_type
        return PartialParsingFrontend(lexer_conf, parser_conf, self.options)

    def __repr__(self):
        return '{}(open({!r}), parser={!r}, lexer={!r}, ...)'.format(type(self).__name__, self.source_path, self.options.parser, self.options.lexer)

    def parse_from_state(self, parse_state: 'PartialParseState', is_end=False):
        return self.parser.parser.parser.parse_from_state(parse_state, is_end=is_end)