from __future__ import annotations
from typing import (
from bson.raw_bson import RawBSONDocument
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import validate_boolean, validate_is_mapping, validate_list
from pymongo.helpers import _gen_index_name, _index_document, _index_list
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
class SearchIndexModel:
    """Represents a search index to create."""
    __slots__ = ('__document',)

    def __init__(self, definition: Mapping[str, Any], name: Optional[str]=None) -> None:
        """Create a Search Index instance.

        For use with :meth:`~pymongo.collection.Collection.create_search_index` and :meth:`~pymongo.collection.Collection.create_search_indexes`.

        :Parameters:
          - `definition` - The definition for this index.
          - `name` (optional) - The name for this index, if present.

        .. versionadded:: 4.5

        .. note:: Search indexes require a MongoDB server version 7.0+ Atlas cluster.
        """
        if name is not None:
            self.__document = dict(name=name, definition=definition)
        else:
            self.__document = dict(definition=definition)

    @property
    def document(self) -> Mapping[str, Any]:
        """The document for this index."""
        return self.__document