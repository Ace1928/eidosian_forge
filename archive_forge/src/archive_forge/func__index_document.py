from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _index_document(index_list: _IndexList) -> SON[str, Any]:
    """Helper to generate an index specifying document.

    Takes a list of (key, direction) pairs.
    """
    if not isinstance(index_list, (list, tuple, abc.Mapping)):
        raise TypeError('must use a dictionary or a list of (key, direction) pairs, not: ' + repr(index_list))
    if not len(index_list):
        raise ValueError('key_or_list must not be empty')
    index: SON[str, Any] = SON()
    if isinstance(index_list, abc.Mapping):
        for key in index_list:
            value = index_list[key]
            _validate_index_key_pair(key, value)
            index[key] = value
    else:
        for item in index_list:
            if isinstance(item, str):
                item = (item, ASCENDING)
            key, value = item
            _validate_index_key_pair(key, value)
            index[key] = value
    return index