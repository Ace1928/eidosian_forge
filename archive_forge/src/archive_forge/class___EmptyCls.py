from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@dataclass(frozen=True)
class __EmptyCls(_BasePattern):

    def to_fsm(self, alphabet=None, prefix_postfix=None, flags=REFlags(0)) -> FSM:
        if alphabet is None:
            alphabet = self.get_alphabet(flags)
        return epsilon(alphabet)

    def _get_alphabet(self, flags: REFlags) -> Alphabet:
        return Alphabet.from_groups({anything_else})

    def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
        return (0, 0)

    def _get_lengths(self) -> Tuple[int, Optional[int]]:
        return (0, 0)

    def simplify(self) -> '__EmptyCls':
        return self