import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def word_num(value: int) -> Iterator[str]:
    if not value:
        yield num_map[value]
    for base, word in num_map.items():
        if base >= 1:
            floor = value // base
            if not floor:
                continue
            elif base >= 100:
                yield from word_num(floor)
                yield ' '
            yield word
            value %= base
            if not value:
                break
            elif base < 100:
                yield '-'
            elif base == 100:
                if lang == 'en':
                    yield ' and '
            else:
                yield ' '