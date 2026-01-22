from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
class ContextualLexer(Lexer):
    lexers: Dict[int, AbstractBasicLexer]
    root_lexer: AbstractBasicLexer
    BasicLexer: Type[AbstractBasicLexer] = BasicLexer

    def __init__(self, conf: 'LexerConf', states: Dict[int, Collection[str]], always_accept: Collection[str]=()) -> None:
        terminals = list(conf.terminals)
        terminals_by_name = conf.terminals_by_name
        trad_conf = copy(conf)
        trad_conf.terminals = terminals
        if has_interegular and (not conf.skip_validation):
            comparator = interegular.Comparator.from_regexes({t: t.pattern.to_regexp() for t in terminals})
        else:
            comparator = None
        lexer_by_tokens: Dict[FrozenSet[str], AbstractBasicLexer] = {}
        self.lexers = {}
        for state, accepts in states.items():
            key = frozenset(accepts)
            try:
                lexer = lexer_by_tokens[key]
            except KeyError:
                accepts = set(accepts) | set(conf.ignore) | set(always_accept)
                lexer_conf = copy(trad_conf)
                lexer_conf.terminals = [terminals_by_name[n] for n in accepts if n in terminals_by_name]
                lexer = self.BasicLexer(lexer_conf, comparator)
                lexer_by_tokens[key] = lexer
            self.lexers[state] = lexer
        assert trad_conf.terminals is terminals
        trad_conf.skip_validation = True
        self.root_lexer = self.BasicLexer(trad_conf, comparator)

    def lex(self, lexer_state: LexerState, parser_state: 'ParserState') -> Iterator[Token]:
        try:
            while True:
                lexer = self.lexers[parser_state.position]
                yield lexer.next_token(lexer_state, parser_state)
        except EOFError:
            pass
        except UnexpectedCharacters as e:
            try:
                last_token = lexer_state.last_token
                token = self.root_lexer.next_token(lexer_state, parser_state)
                raise UnexpectedToken(token, e.allowed, state=parser_state, token_history=[last_token], terminals_by_name=self.root_lexer.terminals_by_name)
            except UnexpectedCharacters:
                raise e