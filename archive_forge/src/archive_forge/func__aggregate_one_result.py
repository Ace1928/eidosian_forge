from __future__ import annotations
from collections import abc
from typing import (
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import ASCENDING, _csot, common, helpers, message
from pymongo.aggregation import (
from pymongo.bulk import _Bulk
from pymongo.change_stream import CollectionChangeStream
from pymongo.collation import validate_collation_or_none
from pymongo.command_cursor import CommandCursor, RawBatchCommandCursor
from pymongo.common import _ecoc_coll_name, _esc_coll_name
from pymongo.cursor import Cursor, RawBatchCursor
from pymongo.errors import (
from pymongo.helpers import _check_write_command_response
from pymongo.message import _UNICODE_REPLACE_CODEC_OPTIONS
from pymongo.operations import (
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.results import (
from pymongo.typings import _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
from pymongo.write_concern import WriteConcern
def _aggregate_one_result(self, conn: Connection, read_preference: Optional[_ServerMode], cmd: SON[str, Any], collation: Optional[_CollationIn], session: Optional[ClientSession]) -> Optional[Mapping[str, Any]]:
    """Internal helper to run an aggregate that returns a single result."""
    result = self._command(conn, cmd, read_preference, allowable_errors=[26], codec_options=self.__write_response_codec_options, read_concern=self.read_concern, collation=collation, session=session)
    if 'cursor' not in result:
        return None
    batch = result['cursor']['firstBatch']
    return batch[0] if batch else None