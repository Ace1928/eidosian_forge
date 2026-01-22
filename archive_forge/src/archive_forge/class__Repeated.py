from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class _Repeated(_BasePattern):
    """Represents a repeated pattern. `base` can be matched from `min` to `max` times.
    `max` may be None to signal infinite"""
    base: _Repeatable
    min: int
    max: Optional[int]

    def __str__(self):
        return f'Repeated[{self.min}:{(self.max if self.max is not None else '')}]:\n{indent(str(self.base), '    ')}'

    def _get_alphabet(self, flags: REFlags) -> Alphabet:
        return self.base.get_alphabet(flags)

    def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
        return self.base.prefix_postfix

    def _get_lengths(self) -> Tuple[int, Optional[int]]:
        l, h = self.base.lengths
        return (l * self.min, h * self.max if None not in (h, self.max) else None)

    def to_fsm(self, alphabet=None, prefix_postfix=None, flags=REFlags(0)) -> FSM:
        if alphabet is None:
            alphabet = self.get_alphabet(flags)
        if prefix_postfix is None:
            prefix_postfix = self.prefix_postfix
        if prefix_postfix != (0, 0):
            raise ValueError('Can not have prefix/postfix on CharGroup-level')
        unit = self.base.to_fsm(alphabet, (0, 0), flags=flags)
        mandatory = unit * self.min
        if self.max is None:
            optional = unit.star()
        else:
            optional = unit.copy()
            optional.__dict__['finals'] |= {optional.initial}
            optional *= self.max - self.min
        return mandatory + optional

    def simplify(self) -> '_Repeated':
        return self.__class__(self.base.simplify(), self.min, self.max)