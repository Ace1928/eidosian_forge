from sys import maxunicode
from typing import cast, Iterable, Iterator, List, MutableSet, Union, Optional
from .unicode_categories import RAW_UNICODE_CATEGORIES
from .codepoints import CodePoint, code_point_order, code_point_repr, \
class RegexError(Exception):
    """
    Error in a regular expression or in a character class specification.
    This exception is derived from `Exception` base class and is raised
    only by the regex subpackage.
    """