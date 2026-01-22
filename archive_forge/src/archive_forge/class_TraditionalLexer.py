import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
class TraditionalLexer(Lexer):

    def __init__(self, terminals, ignore=(), user_callbacks={}, g_regex_flags=0):
        assert all((isinstance(t, TerminalDef) for t in terminals)), terminals
        terminals = list(terminals)
        for t in terminals:
            try:
                re.compile(t.pattern.to_regexp(), g_regex_flags)
            except re.error:
                raise LexError('Cannot compile token %s: %s' % (t.name, t.pattern))
            if t.pattern.min_width == 0:
                raise LexError('Lexer does not allow zero-width terminals. (%s: %s)' % (t.name, t.pattern))
        assert set(ignore) <= {t.name for t in terminals}
        self.newline_types = [t.name for t in terminals if _regexp_has_newline(t.pattern.to_regexp())]
        self.ignore_types = list(ignore)
        terminals.sort(key=lambda x: (-x.priority, -x.pattern.max_width, -len(x.pattern.value), x.name))
        self.terminals = terminals
        self.user_callbacks = user_callbacks
        self.build(g_regex_flags)

    def build(self, g_regex_flags=0):
        terminals, self.callback = _create_unless(self.terminals, g_regex_flags)
        assert all(self.callback.values())
        for type_, f in self.user_callbacks.items():
            if type_ in self.callback:
                self.callback[type_] = CallChain(self.callback[type_], f, lambda t: t.type == type_)
            else:
                self.callback[type_] = f
        self.mres = build_mres(terminals, g_regex_flags)

    def match(self, stream, pos):
        for mre, type_from_index in self.mres:
            m = mre.match(stream, pos)
            if m:
                return (m.group(0), type_from_index[m.lastindex])

    def lex(self, stream):
        return _Lex(self).lex(stream, self.newline_types, self.ignore_types)