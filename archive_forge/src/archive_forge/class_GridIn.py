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
class GridIn:
    """Class to write data to GridFS."""

    def __init__(self, root_collection: Collection, session: Optional[ClientSession]=None, **kwargs: Any) -> None:
        """Write a file to GridFS

        Application developers should generally not need to
        instantiate this class directly - instead see the methods
        provided by :class:`~gridfs.GridFS`.

        Raises :class:`TypeError` if `root_collection` is not an
        instance of :class:`~pymongo.collection.Collection`.

        Any of the file level options specified in the `GridFS Spec
        <http://dochub.mongodb.org/core/gridfsspec>`_ may be passed as
        keyword arguments. Any additional keyword arguments will be
        set as additional fields on the file document. Valid keyword
        arguments include:

          - ``"_id"``: unique ID for this file (default:
            :class:`~bson.objectid.ObjectId`) - this ``"_id"`` must
            not have already been used for another file

          - ``"filename"``: human name for the file

          - ``"contentType"`` or ``"content_type"``: valid mime-type
            for the file

          - ``"chunkSize"`` or ``"chunk_size"``: size of each of the
            chunks, in bytes (default: 255 kb)

          - ``"encoding"``: encoding used for this file. Any :class:`str`
            that is written to the file will be converted to :class:`bytes`.

        :Parameters:
          - `root_collection`: root collection to write to
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession` to use for all
            commands
          - `**kwargs: Any` (optional): file level options (see above)

        .. versionchanged:: 4.0
           Removed the `disable_md5` parameter. See
           :ref:`removed-gridfs-checksum` for details.

        .. versionchanged:: 3.7
           Added the `disable_md5` parameter.

        .. versionchanged:: 3.6
           Added ``session`` parameter.

        .. versionchanged:: 3.0
           `root_collection` must use an acknowledged
           :attr:`~pymongo.collection.Collection.write_concern`
        """
        if not isinstance(root_collection, Collection):
            raise TypeError('root_collection must be an instance of Collection')
        if not root_collection.write_concern.acknowledged:
            raise ConfigurationError('root_collection must use acknowledged write_concern')
        _disallow_transactions(session)
        if 'content_type' in kwargs:
            kwargs['contentType'] = kwargs.pop('content_type')
        if 'chunk_size' in kwargs:
            kwargs['chunkSize'] = kwargs.pop('chunk_size')
        coll = _clear_entity_type_registry(root_collection, read_preference=ReadPreference.PRIMARY)
        kwargs['_id'] = kwargs.get('_id', ObjectId())
        kwargs['chunkSize'] = kwargs.get('chunkSize', DEFAULT_CHUNK_SIZE)
        object.__setattr__(self, '_session', session)
        object.__setattr__(self, '_coll', coll)
        object.__setattr__(self, '_chunks', coll.chunks)
        object.__setattr__(self, '_file', kwargs)
        object.__setattr__(self, '_buffer', io.BytesIO())
        object.__setattr__(self, '_position', 0)
        object.__setattr__(self, '_chunk_number', 0)
        object.__setattr__(self, '_closed', False)
        object.__setattr__(self, '_ensured_index', False)

    def __create_index(self, collection: Collection, index_key: Any, unique: bool) -> None:
        doc = collection.find_one(projection={'_id': 1}, session=self._session)
        if doc is None:
            try:
                index_keys = [index_spec['key'] for index_spec in collection.list_indexes(session=self._session)]
            except OperationFailure:
                index_keys = []
            if index_key not in index_keys:
                collection.create_index(index_key.items(), unique=unique, session=self._session)

    def __ensure_indexes(self) -> None:
        if not object.__getattribute__(self, '_ensured_index'):
            _disallow_transactions(self._session)
            self.__create_index(self._coll.files, _F_INDEX, False)
            self.__create_index(self._coll.chunks, _C_INDEX, True)
            object.__setattr__(self, '_ensured_index', True)

    def abort(self) -> None:
        """Remove all chunks/files that may have been uploaded and close."""
        self._coll.chunks.delete_many({'files_id': self._file['_id']}, session=self._session)
        self._coll.files.delete_one({'_id': self._file['_id']}, session=self._session)
        object.__setattr__(self, '_closed', True)

    @property
    def closed(self) -> bool:
        """Is this file closed?"""
        return self._closed
    _id: Any = _grid_in_property('_id', "The ``'_id'`` value for this file.", read_only=True)
    filename: Optional[str] = _grid_in_property('filename', 'Name of this file.')
    name: Optional[str] = _grid_in_property('filename', 'Alias for `filename`.')
    content_type: Optional[str] = _grid_in_property('contentType', 'DEPRECATED, will be removed in PyMongo 5.0. Mime-type for this file.')
    length: int = _grid_in_property('length', 'Length (in bytes) of this file.', closed_only=True)
    chunk_size: int = _grid_in_property('chunkSize', 'Chunk size for this file.', read_only=True)
    upload_date: datetime.datetime = _grid_in_property('uploadDate', 'Date that this file was uploaded.', closed_only=True)
    md5: Optional[str] = _grid_in_property('md5', 'DEPRECATED, will be removed in PyMongo 5.0. MD5 of the contents of this file if an md5 sum was created.', closed_only=True)
    _buffer: io.BytesIO
    _closed: bool

    def __getattr__(self, name: str) -> Any:
        if name in self._file:
            return self._file[name]
        raise AttributeError("GridIn object has no attribute '%s'" % name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__ or name in self.__class__.__dict__:
            object.__setattr__(self, name, value)
        else:
            self._file[name] = value
            if self._closed:
                self._coll.files.update_one({'_id': self._file['_id']}, {'$set': {name: value}})

    def __flush_data(self, data: Any) -> None:
        """Flush `data` to a chunk."""
        self.__ensure_indexes()
        if not data:
            return
        assert len(data) <= self.chunk_size
        chunk = {'files_id': self._file['_id'], 'n': self._chunk_number, 'data': Binary(data)}
        try:
            self._chunks.insert_one(chunk, session=self._session)
        except DuplicateKeyError:
            self._raise_file_exists(self._file['_id'])
        self._chunk_number += 1
        self._position += len(data)

    def __flush_buffer(self) -> None:
        """Flush the buffer contents out to a chunk."""
        self.__flush_data(self._buffer.getvalue())
        self._buffer.close()
        self._buffer = io.BytesIO()

    def __flush(self) -> Any:
        """Flush the file to the database."""
        try:
            self.__flush_buffer()
            self._file['length'] = Int64(self._position)
            self._file['uploadDate'] = datetime.datetime.now(tz=datetime.timezone.utc)
            return self._coll.files.insert_one(self._file, session=self._session)
        except DuplicateKeyError:
            self._raise_file_exists(self._id)

    def _raise_file_exists(self, file_id: Any) -> NoReturn:
        """Raise a FileExists exception for the given file_id."""
        raise FileExists('file with _id %r already exists' % file_id)

    def close(self) -> None:
        """Flush the file and close it.

        A closed file cannot be written any more. Calling
        :meth:`close` more than once is allowed.
        """
        if not self._closed:
            self.__flush()
            object.__setattr__(self, '_closed', True)

    def read(self, size: int=-1) -> NoReturn:
        raise io.UnsupportedOperation('read')

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def write(self, data: Any) -> None:
        """Write data to the file. There is no return value.

        `data` can be either a string of bytes or a file-like object
        (implementing :meth:`read`). If the file has an
        :attr:`encoding` attribute, `data` can also be a
        :class:`str` instance, which will be encoded as
        :attr:`encoding` before being written.

        Due to buffering, the data may not actually be written to the
        database until the :meth:`close` method is called. Raises
        :class:`ValueError` if this file is already closed. Raises
        :class:`TypeError` if `data` is not an instance of
        :class:`bytes`, a file-like object, or an instance of :class:`str`.
        Unicode data is only allowed if the file has an :attr:`encoding`
        attribute.

        :Parameters:
          - `data`: string of bytes or file-like object to be written
            to the file
        """
        if self._closed:
            raise ValueError('cannot write to a closed file')
        try:
            read = data.read
        except AttributeError:
            if not isinstance(data, (str, bytes)):
                raise TypeError('can only write strings or file-like objects') from None
            if isinstance(data, str):
                try:
                    data = data.encode(self.encoding)
                except AttributeError:
                    raise TypeError('must specify an encoding for file in order to write str') from None
            read = io.BytesIO(data).read
        if self._buffer.tell() > 0:
            space = self.chunk_size - self._buffer.tell()
            if space:
                try:
                    to_write = read(space)
                except BaseException:
                    self.abort()
                    raise
                self._buffer.write(to_write)
                if len(to_write) < space:
                    return
            self.__flush_buffer()
        to_write = read(self.chunk_size)
        while to_write and len(to_write) == self.chunk_size:
            self.__flush_data(to_write)
            to_write = read(self.chunk_size)
        self._buffer.write(to_write)

    def writelines(self, sequence: Iterable[Any]) -> None:
        """Write a sequence of strings to the file.

        Does not add separators.
        """
        for line in sequence:
            self.write(line)

    def writeable(self) -> bool:
        return True

    def __enter__(self) -> GridIn:
        """Support for the context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """Support for the context manager protocol.

        Close the file if no exceptions occur and allow exceptions to propagate.
        """
        if exc_type is None:
            self.close()
        else:
            object.__setattr__(self, '_closed', True)
        return False