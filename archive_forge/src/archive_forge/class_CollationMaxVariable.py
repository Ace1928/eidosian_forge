from __future__ import annotations
from typing import Any, Mapping, Optional, Union
from pymongo import common
class CollationMaxVariable:
    """
    An enum that defines values for `max_variable` on a
    :class:`~pymongo.collation.Collation`.
    """
    PUNCT = 'punct'
    'Both punctuation and spaces are ignored.'
    SPACE = 'space'
    'Spaces alone are ignored.'