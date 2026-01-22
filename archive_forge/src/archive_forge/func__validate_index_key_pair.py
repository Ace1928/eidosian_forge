from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _validate_index_key_pair(key: Any, value: Any) -> None:
    if not isinstance(key, str):
        raise TypeError('first item in each key pair must be an instance of str')
    if not isinstance(value, (str, int, abc.Mapping)):
        raise TypeError("second item in each key pair must be 1, -1, '2d', or another valid MongoDB index specifier.")