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
def __find_and_modify(self, filter: Mapping[str, Any], projection: Optional[Union[Mapping[str, Any], Iterable[str]]], sort: Optional[_IndexList], upsert: Optional[bool]=None, return_document: bool=ReturnDocument.BEFORE, array_filters: Optional[Sequence[Mapping[str, Any]]]=None, hint: Optional[_IndexKeyHint]=None, session: Optional[ClientSession]=None, let: Optional[Mapping]=None, **kwargs: Any) -> Any:
    """Internal findAndModify helper."""
    common.validate_is_mapping('filter', filter)
    if not isinstance(return_document, bool):
        raise ValueError('return_document must be ReturnDocument.BEFORE or ReturnDocument.AFTER')
    collation = validate_collation_or_none(kwargs.pop('collation', None))
    cmd = SON([('findAndModify', self.__name), ('query', filter), ('new', return_document)])
    if let is not None:
        common.validate_is_mapping('let', let)
        cmd['let'] = let
    cmd.update(kwargs)
    if projection is not None:
        cmd['fields'] = helpers._fields_list_to_dict(projection, 'projection')
    if sort is not None:
        cmd['sort'] = helpers._index_document(sort)
    if upsert is not None:
        common.validate_boolean('upsert', upsert)
        cmd['upsert'] = upsert
    if hint is not None:
        if not isinstance(hint, str):
            hint = helpers._index_document(hint)
    write_concern = self._write_concern_for_cmd(cmd, session)

    def _find_and_modify(session: Optional[ClientSession], conn: Connection, retryable_write: bool) -> Any:
        acknowledged = write_concern.acknowledged
        if array_filters is not None:
            if not acknowledged:
                raise ConfigurationError('arrayFilters is unsupported for unacknowledged writes.')
            cmd['arrayFilters'] = list(array_filters)
        if hint is not None:
            if conn.max_wire_version < 8:
                raise ConfigurationError('Must be connected to MongoDB 4.2+ to use hint on find and modify commands.')
            elif not acknowledged and conn.max_wire_version < 9:
                raise ConfigurationError('Must be connected to MongoDB 4.4+ to use hint on unacknowledged find and modify commands.')
            cmd['hint'] = hint
        out = self._command(conn, cmd, read_preference=ReadPreference.PRIMARY, write_concern=write_concern, collation=collation, session=session, retryable_write=retryable_write, user_fields=_FIND_AND_MODIFY_DOC_FIELDS)
        _check_write_command_response(out)
        return out.get('value')
    return self.__database.client._retryable_write(write_concern.acknowledged, _find_and_modify, session)