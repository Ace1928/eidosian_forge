import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \

    Analyze a picture argument of XPath 3.0+ formatting functions.

    :param picture: the picture string.
    :return: a couple of lists containing the literal parts and markers.
    