from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def _combine_char_groups(*groups: _CharGroup, negate):
    pos = set().union(*(g.chars for g in groups if not g.negated))
    neg = set().union(*(g.chars for g in groups if g.negated))
    if neg:
        return _CharGroup(frozenset(neg - pos), not negate)
    else:
        return _CharGroup(frozenset(pos - neg), negate)