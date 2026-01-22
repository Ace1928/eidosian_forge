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
def _aggregation_pipeline(self) -> list[dict[str, Any]]:
    """Return the full aggregation pipeline for this ChangeStream."""
    options = self._change_stream_options()
    full_pipeline: list = [{'$changeStream': options}]
    full_pipeline.extend(self._pipeline)
    return full_pipeline