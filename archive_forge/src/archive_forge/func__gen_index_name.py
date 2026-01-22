from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _gen_index_name(keys: _IndexList) -> str:
    """Generate an index name from the set of fields it is over."""
    return '_'.join(['{}_{}'.format(*item) for item in keys])