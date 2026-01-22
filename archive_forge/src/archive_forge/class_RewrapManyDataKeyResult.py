from __future__ import annotations
import contextlib
import enum
import socket
import weakref
from copy import deepcopy
from typing import (
from bson import _dict_to_bson, decode, encode
from bson.binary import STANDARD, UUID_SUBTYPE, Binary
from bson.codec_options import CodecOptions
from bson.errors import BSONError
from bson.raw_bson import DEFAULT_RAW_BSON_OPTIONS, RawBSONDocument, _inflate_bson
from bson.son import SON
from pymongo import _csot
from pymongo.collection import Collection
from pymongo.common import CONNECT_TIMEOUT
from pymongo.cursor import Cursor
from pymongo.daemon import _spawn_daemon
from pymongo.database import Database
from pymongo.encryption_options import AutoEncryptionOpts, RangeOpts
from pymongo.errors import (
from pymongo.mongo_client import MongoClient
from pymongo.network import BLOCKING_IO_ERRORS
from pymongo.operations import UpdateOne
from pymongo.pool import PoolOptions, _configured_socket, _raise_connection_failure
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, DeleteResult
from pymongo.ssl_support import get_ssl_context
from pymongo.typings import _DocumentType, _DocumentTypeArg
from pymongo.uri_parser import parse_host
from pymongo.write_concern import WriteConcern
class RewrapManyDataKeyResult:
    """Result object returned by a :meth:`~ClientEncryption.rewrap_many_data_key` operation.

    .. versionadded:: 4.2
    """

    def __init__(self, bulk_write_result: Optional[BulkWriteResult]=None) -> None:
        self._bulk_write_result = bulk_write_result

    @property
    def bulk_write_result(self) -> Optional[BulkWriteResult]:
        """The result of the bulk write operation used to update the key vault
        collection with one or more rewrapped data keys. If
        :meth:`~ClientEncryption.rewrap_many_data_key` does not find any matching keys to rewrap,
        no bulk write operation will be executed and this field will be
        ``None``.
        """
        return self._bulk_write_result

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._bulk_write_result!r})'