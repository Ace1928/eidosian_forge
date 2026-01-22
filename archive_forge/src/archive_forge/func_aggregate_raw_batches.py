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
def aggregate_raw_batches(self, pipeline: _Pipeline, session: Optional[ClientSession]=None, comment: Optional[Any]=None, **kwargs: Any) -> RawBatchCursor[_DocumentType]:
    """Perform an aggregation and retrieve batches of raw BSON.

        Similar to the :meth:`aggregate` method but returns a
        :class:`~pymongo.cursor.RawBatchCursor`.

        This example demonstrates how to work with raw batches, but in practice
        raw batches should be passed to an external library that can decode
        BSON into another data type, rather than used with PyMongo's
        :mod:`bson` module.

          >>> import bson
          >>> cursor = db.test.aggregate_raw_batches([
          ...     {'$project': {'x': {'$multiply': [2, '$x']}}}])
          >>> for batch in cursor:
          ...     print(bson.decode_all(batch))

        .. note:: aggregate_raw_batches does not support auto encryption.

        .. versionchanged:: 3.12
           Added session support.

        .. versionadded:: 3.6
        """
    if self.__database.client._encrypter:
        raise InvalidOperation('aggregate_raw_batches does not support auto encryption')
    if comment is not None:
        kwargs['comment'] = comment
    with self.__database.client._tmp_session(session, close=False) as s:
        return cast(RawBatchCursor[_DocumentType], self._aggregate(_CollectionRawAggregationCommand, pipeline, RawBatchCommandCursor, session=s, explicit_session=session is not None, **kwargs))