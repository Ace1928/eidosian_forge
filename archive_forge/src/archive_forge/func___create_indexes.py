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
@_csot.apply
def __create_indexes(self, indexes: Sequence[IndexModel], session: Optional[ClientSession], **kwargs: Any) -> list[str]:
    """Internal createIndexes helper.

        :Parameters:
          - `indexes`: A list of :class:`~pymongo.operations.IndexModel`
            instances.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`.
          - `**kwargs` (optional): optional arguments to the createIndexes
            command (like maxTimeMS) can be passed as keyword arguments.
        """
    names = []
    with self._conn_for_writes(session) as conn:
        supports_quorum = conn.max_wire_version >= 9

        def gen_indexes() -> Iterator[Mapping[str, Any]]:
            for index in indexes:
                if not isinstance(index, IndexModel):
                    raise TypeError(f'{index!r} is not an instance of pymongo.operations.IndexModel')
                document = index.document
                names.append(document['name'])
                yield document
        cmd = SON([('createIndexes', self.name), ('indexes', list(gen_indexes()))])
        cmd.update(kwargs)
        if 'commitQuorum' in kwargs and (not supports_quorum):
            raise ConfigurationError('Must be connected to MongoDB 4.4+ to use the commitQuorum option for createIndexes')
        self._command(conn, cmd, read_preference=ReadPreference.PRIMARY, codec_options=_UNICODE_REPLACE_CODEC_OPTIONS, write_concern=self._write_concern_for(session), session=session)
    return names