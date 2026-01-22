from __future__ import annotations
import datetime
import io
import math
import os
from typing import Any, Iterable, Mapping, NoReturn, Optional
from bson.binary import Binary
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.son import SON
from gridfs.errors import CorruptGridFile, FileExists, NoFile
from pymongo import ASCENDING
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import (
from pymongo.read_preferences import ReadPreference
def _grid_out_property(field_name: str, docstring: str) -> Any:
    """Create a GridOut property."""

    def getter(self: Any) -> Any:
        self._ensure_file()
        if field_name == 'length':
            return self._file.get(field_name, 0)
        return self._file.get(field_name, None)
    docstring += '\n\nThis attribute is read-only.'
    return property(getter, doc=docstring)