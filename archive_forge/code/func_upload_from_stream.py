from __future__ import annotations
from collections import abc
from typing import Any, Mapping, Optional, cast
from bson.objectid import ObjectId
from gridfs.errors import NoFile
from gridfs.grid_file import (
from pymongo import ASCENDING, DESCENDING, _csot
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.common import validate_string
from pymongo.database import Database
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import _ServerMode
from pymongo.write_concern import WriteConcern
@_csot.apply
def upload_from_stream(self, filename: str, source: Any, chunk_size_bytes: Optional[int]=None, metadata: Optional[Mapping[str, Any]]=None, session: Optional[ClientSession]=None) -> ObjectId:
    """Uploads a user file to a GridFS bucket.

        Reads the contents of the user file from `source` and uploads
        it to the file `filename`. Source can be a string or file-like object.
        For example::

          my_db = MongoClient().test
          fs = GridFSBucket(my_db)
          file_id = fs.upload_from_stream(
              "test_file",
              "data I want to store!",
              chunk_size_bytes=4,
              metadata={"contentType": "text/plain"})

        Returns the _id of the uploaded file.

        Raises :exc:`~gridfs.errors.NoFile` if no such version of
        that file exists.
        Raises :exc:`~ValueError` if `filename` is not a string.

        :Parameters:
          - `filename`: The name of the file to upload.
          - `source`: The source stream of the content to be uploaded. Must be
            a file-like object that implements :meth:`read` or a string.
          - `chunk_size_bytes` (options): The number of bytes per chunk of this
            file. Defaults to the chunk_size_bytes of :class:`GridFSBucket`.
          - `metadata` (optional): User data for the 'metadata' field of the
            files collection document. If not provided the metadata field will
            be omitted from the files collection document.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
    with self.open_upload_stream(filename, chunk_size_bytes, metadata, session=session) as gin:
        gin.write(source)
    return cast(ObjectId, gin._id)