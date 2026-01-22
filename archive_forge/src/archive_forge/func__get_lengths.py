from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def _get_lengths(self) -> Tuple[int, Optional[int]]:
    low, high = (None, 0)
    for o in self.options:
        ol, oh = o.lengths
        if low is None or ol < low:
            low = ol
        if oh is None or (high is not None and oh > high):
            high = oh
    return (low, high)