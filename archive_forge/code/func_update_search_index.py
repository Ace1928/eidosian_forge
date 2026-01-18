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
def update_search_index(self, name: str, definition: Mapping[str, Any], session: Optional[ClientSession]=None, comment: Optional[Any]=None, **kwargs: Any) -> None:
    """Update a search index by replacing the existing index definition with the provided definition.

        :Parameters:
          - `name`: The name of the search index to be updated.
          - `definition`: The new search index definition.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `comment` (optional): A user-provided comment to attach to this
            command.
          - `**kwargs` (optional): optional arguments to the updateSearchIndexes
            command (like maxTimeMS) can be passed as keyword arguments.

        .. note:: requires a MongoDB server version 7.0+ Atlas cluster.

        .. versionadded:: 4.5
        """
    cmd = SON([('updateSearchIndex', self.__name), ('name', name), ('definition', definition)])
    cmd.update(kwargs)
    if comment is not None:
        cmd['comment'] = comment
    with self._conn_for_writes(session) as conn:
        self._command(conn, cmd, read_preference=ReadPreference.PRIMARY, allowable_errors=['ns not found', 26], codec_options=_UNICODE_REPLACE_CODEC_OPTIONS)