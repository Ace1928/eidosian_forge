from __future__ import annotations
import datetime
import io
import math
import os
from typing import Any, Iterable, Mapping, NoReturn, Optional
from bson.binary import Binary
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.son import SON
from gridfs.errors import CorruptGridFile, FileExists, NoFile
from pymongo import ASCENDING
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import (
from pymongo.read_preferences import ReadPreference
class GridOutCursor(Cursor):
    """A cursor / iterator for returning GridOut objects as the result
    of an arbitrary query against the GridFS files collection.
    """

    def __init__(self, collection: Collection, filter: Optional[Mapping[str, Any]]=None, skip: int=0, limit: int=0, no_cursor_timeout: bool=False, sort: Optional[Any]=None, batch_size: int=0, session: Optional[ClientSession]=None) -> None:
        """Create a new cursor, similar to the normal
        :class:`~pymongo.cursor.Cursor`.

        Should not be called directly by application developers - see
        the :class:`~gridfs.GridFS` method :meth:`~gridfs.GridFS.find` instead.

        .. versionadded 2.7

        .. seealso:: The MongoDB documentation on `cursors <https://dochub.mongodb.org/core/cursors>`_.
        """
        _disallow_transactions(session)
        collection = _clear_entity_type_registry(collection)
        self.__root_collection = collection
        super().__init__(collection.files, filter, skip=skip, limit=limit, no_cursor_timeout=no_cursor_timeout, sort=sort, batch_size=batch_size, session=session)

    def next(self) -> GridOut:
        """Get next GridOut object from cursor."""
        _disallow_transactions(self.session)
        next_file = super().next()
        return GridOut(self.__root_collection, file_document=next_file, session=self.session)
    __next__ = next

    def add_option(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError('Method does not exist for GridOutCursor')

    def remove_option(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError('Method does not exist for GridOutCursor')

    def _clone_base(self, session: Optional[ClientSession]) -> GridOutCursor:
        """Creates an empty GridOutCursor for information to be copied into."""
        return GridOutCursor(self.__root_collection, session=session)