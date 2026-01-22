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
class BasicLexer(AbstractBasicLexer):
    terminals: Collection[TerminalDef]
    ignore_types: FrozenSet[str]
    newline_types: FrozenSet[str]
    user_callbacks: Dict[str, _Callback]
    callback: Dict[str, _Callback]
    re: ModuleType

    def __init__(self, conf: 'LexerConf', comparator=None) -> None:
        terminals = list(conf.terminals)
        assert all((isinstance(t, TerminalDef) for t in terminals)), terminals
        self.re = conf.re_module
        if not conf.skip_validation:
            terminal_to_regexp = {}
            for t in terminals:
                regexp = t.pattern.to_regexp()
                try:
                    self.re.compile(regexp, conf.g_regex_flags)
                except self.re.error:
                    raise LexError('Cannot compile token %s: %s' % (t.name, t.pattern))
                if t.pattern.min_width == 0:
                    raise LexError('Lexer does not allow zero-width terminals. (%s: %s)' % (t.name, t.pattern))
                if t.pattern.type == 're':
                    terminal_to_regexp[t] = regexp
            if not set(conf.ignore) <= {t.name for t in terminals}:
                raise LexError('Ignore terminals are not defined: %s' % (set(conf.ignore) - {t.name for t in terminals}))
            if has_interegular:
                _check_regex_collisions(terminal_to_regexp, comparator, conf.strict)
            elif conf.strict:
                raise LexError("interegular must be installed for strict mode. Use `pip install 'lark[interegular]'`.")
        self.newline_types = frozenset((t.name for t in terminals if _regexp_has_newline(t.pattern.to_regexp())))
        self.ignore_types = frozenset(conf.ignore)
        terminals.sort(key=lambda x: (-x.priority, -x.pattern.max_width, -len(x.pattern.value), x.name))
        self.terminals = terminals
        self.user_callbacks = conf.callbacks
        self.g_regex_flags = conf.g_regex_flags
        self.use_bytes = conf.use_bytes
        self.terminals_by_name = conf.terminals_by_name
        self._scanner = None

    def _build_scanner(self):
        terminals, self.callback = _create_unless(self.terminals, self.g_regex_flags, self.re, self.use_bytes)
        assert all(self.callback.values())
        for type_, f in self.user_callbacks.items():
            if type_ in self.callback:
                self.callback[type_] = CallChain(self.callback[type_], f, lambda t: t.type == type_)
            else:
                self.callback[type_] = f
        self._scanner = Scanner(terminals, self.g_regex_flags, self.re, self.use_bytes)

    @property
    def scanner(self):
        if self._scanner is None:
            self._build_scanner()
        return self._scanner

    def match(self, text, pos):
        return self.scanner.match(text, pos)

    def next_token(self, lex_state: LexerState, parser_state: Any=None) -> Token:
        line_ctr = lex_state.line_ctr
        while line_ctr.char_pos < len(lex_state.text):
            res = self.match(lex_state.text, line_ctr.char_pos)
            if not res:
                allowed = self.scanner.allowed_types - self.ignore_types
                if not allowed:
                    allowed = {'<END-OF-FILE>'}
                raise UnexpectedCharacters(lex_state.text, line_ctr.char_pos, line_ctr.line, line_ctr.column, allowed=allowed, token_history=lex_state.last_token and [lex_state.last_token], state=parser_state, terminals_by_name=self.terminals_by_name)
            value, type_ = res
            ignored = type_ in self.ignore_types
            t = None
            if not ignored or type_ in self.callback:
                t = Token(type_, value, line_ctr.char_pos, line_ctr.line, line_ctr.column)
            line_ctr.feed(value, type_ in self.newline_types)
            if t is not None:
                t.end_line = line_ctr.line
                t.end_column = line_ctr.column
                t.end_pos = line_ctr.char_pos
                if t.type in self.callback:
                    t = self.callback[t.type](t)
                if not ignored:
                    if not isinstance(t, Token):
                        raise LexError('Callbacks must return a token (returned %r)' % t)
                    lex_state.last_token = t
                    return t
        raise EOFError(self)