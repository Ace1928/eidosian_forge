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
@classmethod
def _handle_prepositional_phrase(cls, phrase, transform, sep):
    """
        Given a word or phrase possibly separated by sep, parse out
        the prepositional phrase and apply the transform to the word
        preceding the prepositional phrase.

        Raise ValueError if the pivot is not found or if at least two
        separators are not found.

        >>> engine._handle_prepositional_phrase("man-of-war", str.upper, '-')
        'MAN-of-war'
        >>> engine._handle_prepositional_phrase("man of war", str.upper, ' ')
        'MAN of war'
        """
    parts = phrase.split(sep)
    if len(parts) < 3:
        raise ValueError('Cannot handle words with fewer than two separators')
    pivot = cls._find_pivot(parts, pl_prep_list_da)
    transformed = transform(parts[pivot - 1]) or parts[pivot - 1]
    return ' '.join(parts[:pivot - 1] + [sep.join([transformed, parts[pivot], ''])]) + ' '.join(parts[pivot + 1:])