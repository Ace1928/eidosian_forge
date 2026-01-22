from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, Type, Union
from bson import CodecOptions, _bson_to_dict
from bson.raw_bson import RawBSONDocument
from bson.timestamp import Timestamp
from pymongo import _csot, common
from pymongo.aggregation import (
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
def _resumable(exc: PyMongoError) -> bool:
    """Return True if given a resumable change stream error."""
    if isinstance(exc, (ConnectionFailure, CursorNotFound)):
        return True
    if isinstance(exc, OperationFailure):
        if exc._max_wire_version is None:
            return False
        return exc._max_wire_version >= 9 and exc.has_error_label('ResumableChangeStreamError') or (exc._max_wire_version < 9 and exc.code in _RESUMABLE_GETMORE_ERRORS)
    return False