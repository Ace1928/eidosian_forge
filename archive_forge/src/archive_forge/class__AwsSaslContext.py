from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
class _AwsSaslContext(AwsSaslContext):

    def binary_type(self) -> Type[Binary]:
        """Return the bson.binary.Binary type."""
        return Binary

    def bson_encode(self, doc: Mapping[str, Any]) -> bytes:
        """Encode a dictionary to BSON."""
        return bson.encode(doc)

    def bson_decode(self, data: _ReadableBuffer) -> Mapping[str, Any]:
        """Decode BSON to a dictionary."""
        return bson.decode(data)