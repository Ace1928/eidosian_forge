from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
class _ParsePattern(SimpleParser[Pattern]):
    SPECIAL_CHARS_STANDARD: FrozenSet[str] = frozenset({'+', '?', '*', '.', '$', '^', '\\', '(', ')', '[', '|'})
    SPECIAL_CHARS_INNER: FrozenSet[str] = frozenset({'\\', ']'})
    RESERVED_ESCAPES: FrozenSet[str] = frozenset({'u', 'U', 'A', 'Z', 'b', 'B'})

    def __init__(self, data: str):
        super(_ParsePattern, self).__init__(data)
        self.flags = None

    def parse(self):
        try:
            return super(_ParsePattern, self).parse()
        except NoMatch:
            raise InvalidSyntax

    def start(self):
        self.flags = None
        p = self.pattern()
        if self.flags is not None:
            p = p.with_flags(self.flags)
        return p

    def pattern(self):
        options = [self.conc()]
        while self.static_b('|'):
            options.append(self.conc())
        return Pattern(tuple(options))

    def conc(self):
        parts = []
        while True:
            try:
                parts.append(self.obj())
            except nomatch:
                break
        return _Concatenation(tuple(parts))

    def obj(self):
        if self.static_b('('):
            return self.group()
        return self.repetition(self.atom())

    def group(self):
        if self.static_b('?'):
            return self.extension_group()
        else:
            p = self.pattern()
            self.static(')')
            return self.repetition(p)

    def extension_group(self):
        c = self.any()
        if c in 'aiLmsux-':
            self.index -= 1
            added_flags = self.multiple('aiLmsux', 0, None)
            if self.static_b('-'):
                removed_flags = self.multiple('aiLmsux', 1, None)
            else:
                removed_flags = ''
            if self.static_b(':'):
                p = self.pattern()
                p = p.with_flags(_get_flags(added_flags), _get_flags(removed_flags))
                self.static(')')
                return self.repetition(p)
            elif removed_flags != '':
                raise nomatch
            else:
                self.static(')')
                self.flags = _get_flags(added_flags)
                return _EMPTY
        elif c == ':':
            p = self.pattern()
            self.static(')')
            return self.repetition(p)
        elif c == 'P':
            if self.static_b('<'):
                self.multiple('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_', 1, None)
                self.static('>')
                p = self.pattern()
                self.static(')')
                return self.repetition(p)
            elif self.static_b('='):
                raise Unsupported('Group references are not implemented')
        elif c == '#':
            while not self.static_b(')'):
                self.any()
        elif c == '=':
            p = self.pattern()
            self.static(')')
            return _NonCapturing(p, False, False)
        elif c == '!':
            p = self.pattern()
            self.static(')')
            return _NonCapturing(p, False, True)
        elif c == '<':
            c = self.any()
            if c == '=':
                p = self.pattern()
                self.static(')')
                return _NonCapturing(p, True, False)
            elif c == '!':
                p = self.pattern()
                self.static(')')
                return _NonCapturing(p, True, True)
        elif c == '(':
            raise Unsupported('Conditional matching is not implemented')
        else:
            raise InvalidSyntax(f'Unknown group-extension: {c!r} (Context: {self.data[self.index - 3:self.index + 5]!r}')

    def atom(self):
        if self.static_b('['):
            return self.repetition(self.chargroup())
        elif self.static_b('\\'):
            return self.repetition(self.escaped())
        elif self.static_b('.'):
            return self.repetition(_DOT)
        elif self.static_b('$'):
            raise Unsupported("'$'")
        elif self.static_b('^'):
            raise Unsupported("'^'")
        else:
            c = self.any_but(*self.SPECIAL_CHARS_STANDARD)
            return self.repetition(_CharGroup(frozenset({c}), False))

    def repetition(self, base: _Repeatable):
        if self.static_b('*'):
            if self.static_b('?'):
                pass
            return _Repeated(base, 0, None)
        elif self.static_b('+'):
            if self.static_b('?'):
                pass
            return _Repeated(base, 1, None)
        elif self.static_b('?'):
            if self.static_b('?'):
                pass
            return _Repeated(base, 0, 1)
        elif self.static_b('{'):
            try:
                n = self.number()
            except nomatch:
                n = 0
            if self.static_b(','):
                try:
                    m = self.number()
                except nomatch:
                    m = None
            else:
                m = n
            self.static('}')
            if self.static_b('?'):
                pass
            return _Repeated(base, n, m)
        else:
            return base

    def number(self) -> int:
        return int(self.multiple('0123456789', 1, None))

    def escaped(self, inner=False):
        if self.static_b('x'):
            n = self.multiple('0123456789abcdefABCDEF', 2, 2)
            c = chr(int(n, 16))
            return _CharGroup(frozenset({c}), False)
        if self.static_b('0'):
            n = self.multiple('01234567', 1, 2)
            c = chr(int(n, 8))
            return _CharGroup(frozenset({c}), False)
        if self.anyof_b('N', 'p', 'P', 'u', 'U'):
            raise Unsupported('regex module unicode properties are not supported.')
        if not inner:
            try:
                n = self.multiple('01234567', 3, 3)
            except nomatch:
                pass
            else:
                c = chr(int(n, 8))
                return _CharGroup(frozenset({c}), False)
            try:
                self.multiple('0123456789', 1, 2)
            except nomatch:
                pass
            else:
                raise Unsupported('Group references are not implemented')
        else:
            try:
                n = self.multiple('01234567', 1, 3)
            except nomatch:
                pass
            else:
                c = chr(int(n, 8))
                return _CharGroup(frozenset({c}), False)
        if not inner:
            try:
                c = self.anyof(*self.RESERVED_ESCAPES)
            except nomatch:
                pass
            else:
                raise Unsupported(f'Escape \\{c} is not implemented')
        try:
            c = self.anyof(*_CHAR_GROUPS)
        except nomatch:
            pass
        else:
            return _CHAR_GROUPS[c]
        c = self.any_but('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if c.isalpha():
            raise nomatch
        return _CharGroup(frozenset(c), False)

    def chargroup(self):
        if self.static_b('^'):
            negate = True
        else:
            negate = False
        groups = []
        while True:
            try:
                groups.append(self.chargroup_inner())
            except nomatch:
                break
        self.static(']')
        if len(groups) == 1:
            f = tuple(groups)[0]
            return _CharGroup(f.chars, negate ^ f.negated)
        elif len(groups) == 0:
            return _CharGroup(frozenset({}), negate)
        else:
            return _combine_char_groups(*groups, negate=negate)

    def chargroup_inner(self) -> _CharGroup:
        start = self.index
        if self.static_b('\\'):
            base = self.escaped(True)
        else:
            base = _CharGroup(frozenset(self.any_but(*self.SPECIAL_CHARS_INNER)), False)
        if self.static_b('-'):
            if self.static_b('\\'):
                end = self.escaped(True)
            elif self.peek_static(']'):
                return _combine_char_groups(base, _CharGroup(frozenset('-'), False), negate=False)
            else:
                end = _CharGroup(frozenset(self.any_but(*self.SPECIAL_CHARS_INNER)), False)
            if len(base.chars) != 1 or len(end.chars) != 1:
                raise InvalidSyntax(f'Invalid Character-range: {self.data[start:self.index]}')
            low, high = (ord(*base.chars), ord(*end.chars))
            if low > high:
                raise InvalidSyntax(f'Invalid Character-range: {self.data[start:self.index]}')
            return _CharGroup(frozenset((chr(i) for i in range(low, high + 1))), False)
        return base