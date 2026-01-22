from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def _get_alphabet(self, flags: REFlags) -> Alphabet:
    flags = _combine_flags(flags, self.added_flags, self.removed_flags)
    return Alphabet.union(*(p.get_alphabet(flags) for p in self.options))[0]