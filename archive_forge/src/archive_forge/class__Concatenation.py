from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class _Concatenation(_BasePattern):
    """Represents multiple Patterns that have to be match in a row. Can contain `_NonCapturing`"""
    parts: Tuple[Union[_BasePattern, _NonCapturing], ...]
    __slots__ = ('parts',)

    def __str__(self):
        return 'Concatenation:\n' + '\n'.join((indent(str(p), '  ') for p in self.parts))

    def _get_alphabet(self, flags: REFlags) -> Alphabet:
        return Alphabet.union(*(p.get_alphabet(flags) for p in self.parts))[0]

    def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
        pre = 0
        off = 0
        for p in self.parts:
            if not isinstance(p, _NonCapturing):
                off += p.lengths[0]
            elif p.backwards:
                a, b = p.inner.lengths
                if a != b:
                    raise InvalidSyntax(f'lookbacks have to have fixed length {(a, b)}')
                req = a - off
                if req > pre:
                    pre = req
        post = 0
        off = 0
        for p in reversed(self.parts):
            if not isinstance(p, _NonCapturing):
                off += p.lengths[0]
            elif not p.backwards:
                a, b = p.inner.lengths
                if b is None:
                    req = a - off
                else:
                    req = b - off
                if req > post:
                    post = req
        return (pre, post)

    def _get_lengths(self) -> Tuple[int, Optional[int]]:
        low, high = (0, 0)
        for p in self.parts:
            if not isinstance(p, _NonCapturing):
                pl, ph = p.lengths
                low += pl
                high = high + ph if None not in (high, ph) else None
        return (low, high)

    def to_fsm(self, alphabet=None, prefix_postfix=None, flags=REFlags(0)) -> FSM:
        if alphabet is None:
            alphabet = self.get_alphabet(flags)
        if prefix_postfix is None:
            prefix_postfix = self.prefix_postfix
        if prefix_postfix[0] < self.prefix_postfix[0] or prefix_postfix[1] < self.prefix_postfix[1]:
            raise Unsupported('Group can not have lookbacks/lookaheads that go beyond the group bounds.')
        all_ = _ALL.to_fsm(alphabet)
        all_star = all_.star()
        fsm_parts = []
        current = [all_.times(prefix_postfix[0])]
        for part in self.parts:
            if isinstance(part, _NonCapturing):
                inner = part.inner.to_fsm(alphabet, (0, 0), flags)
                if part.backwards:
                    raise Unsupported('lookbacks are not implemented')
                else:
                    fsm_parts.append((None, current))
                    fsm_parts.append((part, inner))
                    current = []
            else:
                current.append(part.to_fsm(alphabet, (0, 0), flags))
        current.append(all_.times(prefix_postfix[1]))
        result = FSM.concatenate(*current)
        for m, f in reversed(fsm_parts):
            if m is None:
                result = FSM.concatenate(*f, result)
            else:
                assert isinstance(m, _NonCapturing) and (not m.backwards)
                if m.negate:
                    result = result.difference(f + all_star)
                else:
                    result = result.intersection(f + all_star)
        return result

    def simplify(self) -> '_Concatenation':
        return self.__class__(tuple((p.simplify() for p in self.parts)))