from sys import maxunicode
from typing import cast, Iterable, Iterator, List, MutableSet, Union, Optional
from .unicode_categories import RAW_UNICODE_CATEGORIES
from .codepoints import CodePoint, code_point_order, code_point_repr, \
def unicode_subset(name: str) -> UnicodeSubset:
    if name.startswith('Is'):
        try:
            return UNICODE_BLOCKS[name]
        except KeyError:
            raise RegexError("%r doesn't match to any Unicode block." % name)
    else:
        try:
            return UNICODE_CATEGORIES[name]
        except KeyError:
            raise RegexError("%r doesn't match to any Unicode category." % name)