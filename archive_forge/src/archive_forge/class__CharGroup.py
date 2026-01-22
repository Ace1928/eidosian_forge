from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class _CharGroup(_Repeatable):
    """Represents the smallest possible pattern that can be matched: A single char.
    Direct port from the lego module"""
    chars: FrozenSet[str]
    negated: bool
    __slots__ = ('chars', 'negated')

    def _get_alphabet(self, flags: REFlags) -> Alphabet:
        if flags & REFlags.CASE_INSENSITIVE:
            relevant = {*map(str.lower, self.chars), *map(str.upper, self.chars)}
        else:
            relevant = self.chars
        return Alphabet.from_groups(relevant, {anything_else})

    def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
        return (0, 0)

    def _get_lengths(self) -> Tuple[int, Optional[int]]:
        return (1, 1)

    def to_fsm(self, alphabet=None, prefix_postfix=None, flags=REFlags(0)) -> FSM:
        if alphabet is None:
            alphabet = self.get_alphabet(flags)
        if prefix_postfix is None:
            prefix_postfix = self.prefix_postfix
        if prefix_postfix != (0, 0):
            raise ValueError('Can not have prefix/postfix on CharGroup-level')
        insensitive = flags & REFlags.CASE_INSENSITIVE
        flags &= ~REFlags.CASE_INSENSITIVE
        flags &= ~REFlags.SINGLE_LINE
        if flags:
            raise Unsupported(flags)
        if insensitive:
            chars = frozenset({*(c.lower() for c in self.chars), *(c.upper() for c in self.chars)})
        else:
            chars = self.chars
        if self.negated:
            mapping = {0: {alphabet[symbol]: 1 for symbol in set(alphabet) - chars}}
        else:
            mapping = {0: {alphabet[symbol]: 1 for symbol in chars}}
        return FSM(alphabet=alphabet, states={0, 1}, initial=0, finals={1}, map=mapping)

    def simplify(self) -> '_CharGroup':
        return self