from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def _get_prefix_postfix(self) -> Tuple[int, Optional[int]]:
    pre, post = (0, 0)
    for o in self.options:
        opre, opost = o.prefix_postfix
        if opre > pre:
            pre = opre
        if opost is None or (post is not None and opost > post):
            post = opost
    return (pre, post)