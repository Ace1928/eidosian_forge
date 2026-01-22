from __future__ import annotations
from collections import deque
from typing import (
from bson import CodecOptions, _convert_raw_document_lists_to_streams
from pymongo.cursor import _CURSOR_CLOSED_ERRORS, _ConnectionManager
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.message import _CursorAddress, _GetMore, _OpMsg, _OpReply, _RawBatchGetMore
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _DocumentOut, _DocumentType
class RawBatchCommandCursor(CommandCursor, Generic[_DocumentType]):
    _getmore_class = _RawBatchGetMore

    def __init__(self, collection: Collection[_DocumentType], cursor_info: Mapping[str, Any], address: Optional[_Address], batch_size: int=0, max_await_time_ms: Optional[int]=None, session: Optional[ClientSession]=None, explicit_session: bool=False, comment: Any=None) -> None:
        """Create a new cursor / iterator over raw batches of BSON data.

        Should not be called directly by application developers -
        see :meth:`~pymongo.collection.Collection.aggregate_raw_batches`
        instead.

        .. seealso:: The MongoDB documentation on `cursors <https://dochub.mongodb.org/core/cursors>`_.
        """
        assert not cursor_info.get('firstBatch')
        super().__init__(collection, cursor_info, address, batch_size, max_await_time_ms, session, explicit_session, comment)

    def _unpack_response(self, response: Union[_OpReply, _OpMsg], cursor_id: Optional[int], codec_options: CodecOptions, user_fields: Optional[Mapping[str, Any]]=None, legacy_response: bool=False) -> list[Mapping[str, Any]]:
        raw_response = response.raw_response(cursor_id, user_fields=user_fields)
        if not legacy_response:
            _convert_raw_document_lists_to_streams(raw_response[0])
        return raw_response

    def __getitem__(self, index: int) -> NoReturn:
        raise InvalidOperation('Cannot call __getitem__ on RawBatchCursor')