import collections
import itertools
import re
from typing import Any, Callable, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
def _parse_letter_version(letter: str, number: Union[str, bytes, SupportsInt]) -> Optional[Tuple[str, int]]:
    if letter:
        if number is None:
            number = 0
        letter = letter.lower()
        if letter == 'alpha':
            letter = 'a'
        elif letter == 'beta':
            letter = 'b'
        elif letter in ['c', 'pre', 'preview']:
            letter = 'rc'
        elif letter in ['rev', 'r']:
            letter = 'post'
        return (letter, int(number))
    if not letter and number:
        letter = 'post'
        return (letter, int(number))
    return None