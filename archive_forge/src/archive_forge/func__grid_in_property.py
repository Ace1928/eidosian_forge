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
def _grid_in_property(field_name: str, docstring: str, read_only: Optional[bool]=False, closed_only: Optional[bool]=False) -> Any:
    """Create a GridIn property."""

    def getter(self: Any) -> Any:
        if closed_only and (not self._closed):
            raise AttributeError('can only get %r on a closed file' % field_name)
        if field_name == 'length':
            return self._file.get(field_name, 0)
        return self._file.get(field_name, None)

    def setter(self: Any, value: Any) -> Any:
        if self._closed:
            self._coll.files.update_one({'_id': self._file['_id']}, {'$set': {field_name: value}})
        self._file[field_name] = value
    if read_only:
        docstring += '\n\nThis attribute is read-only.'
    elif closed_only:
        docstring = '{}\n\n{}'.format(docstring, 'This attribute is read-only and can only be read after :meth:`close` has been called.')
    if not read_only and (not closed_only):
        return property(getter, setter, doc=docstring)
    return property(getter, doc=docstring)