from __future__ import annotations
from typing import (
from bson.raw_bson import RawBSONDocument
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import validate_boolean, validate_is_mapping, validate_list
from pymongo.helpers import _gen_index_name, _index_document, _index_list
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
class InsertOne(Generic[_DocumentType]):
    """Represents an insert_one operation."""
    __slots__ = ('_doc',)

    def __init__(self, document: _DocumentType) -> None:
        """Create an InsertOne instance.

        For use with :meth:`~pymongo.collection.Collection.bulk_write`.

        :Parameters:
          - `document`: The document to insert. If the document is missing an
            _id field one will be added.
        """
        self._doc = document

    def _add_to_bulk(self, bulkobj: _Bulk) -> None:
        """Add this operation to the _Bulk instance `bulkobj`."""
        bulkobj.add_insert(self._doc)

    def __repr__(self) -> str:
        return f'InsertOne({self._doc!r})'

    def __eq__(self, other: Any) -> bool:
        if type(other) == type(self):
            return other._doc == self._doc
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other