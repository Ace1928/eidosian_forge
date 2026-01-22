import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def _handle_long_compounds(self, word: Words, count: int) -> Union[str, None]:
    """
        Handles the plural and singular for compound `Words` that
        have three or more words, based on the given count.

        >>> engine()._handle_long_compounds(Words("pair of scissors"), 2)
        'pairs of scissors'
        >>> engine()._handle_long_compounds(Words("men beyond hills"), 1)
        'man beyond hills'
        """
    inflection = self._sinoun if count == 1 else self._plnoun
    solutions = (' '.join(itertools.chain(leader, [inflection(cand, count), prep], trailer)) for leader, (cand, prep), trailer in windowed_complete(word.split_, 2) if prep in pl_prep_list_da)
    return next(solutions, None)