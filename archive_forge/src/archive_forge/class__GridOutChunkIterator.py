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
class _GridOutChunkIterator:
    """Iterates over a file's chunks using a single cursor.

    Raises CorruptGridFile when encountering any truncated, missing, or extra
    chunk in a file.
    """

    def __init__(self, grid_out: GridOut, chunks: Collection, session: Optional[ClientSession], next_chunk: Any) -> None:
        self._id = grid_out._id
        self._chunk_size = int(grid_out.chunk_size)
        self._length = int(grid_out.length)
        self._chunks = chunks
        self._session = session
        self._next_chunk = next_chunk
        self._num_chunks = math.ceil(float(self._length) / self._chunk_size)
        self._cursor = None
    _cursor: Optional[Cursor]

    def expected_chunk_length(self, chunk_n: int) -> int:
        if chunk_n < self._num_chunks - 1:
            return self._chunk_size
        return self._length - self._chunk_size * (self._num_chunks - 1)

    def __iter__(self) -> _GridOutChunkIterator:
        return self

    def _create_cursor(self) -> None:
        filter = {'files_id': self._id}
        if self._next_chunk > 0:
            filter['n'] = {'$gte': self._next_chunk}
        _disallow_transactions(self._session)
        self._cursor = self._chunks.find(filter, sort=[('n', 1)], session=self._session)

    def _next_with_retry(self) -> Mapping[str, Any]:
        """Return the next chunk and retry once on CursorNotFound.

        We retry on CursorNotFound to maintain backwards compatibility in
        cases where two calls to read occur more than 10 minutes apart (the
        server's default cursor timeout).
        """
        if self._cursor is None:
            self._create_cursor()
            assert self._cursor is not None
        try:
            return self._cursor.next()
        except CursorNotFound:
            self._cursor.close()
            self._create_cursor()
            return self._cursor.next()

    def next(self) -> Mapping[str, Any]:
        try:
            chunk = self._next_with_retry()
        except StopIteration:
            if self._next_chunk >= self._num_chunks:
                raise
            raise CorruptGridFile('no chunk #%d' % self._next_chunk) from None
        if chunk['n'] != self._next_chunk:
            self.close()
            raise CorruptGridFile('Missing chunk: expected chunk #%d but found chunk with n=%d' % (self._next_chunk, chunk['n']))
        if chunk['n'] >= self._num_chunks:
            if len(chunk['data']):
                self.close()
                raise CorruptGridFile('Extra chunk found: expected %d chunks but found chunk with n=%d' % (self._num_chunks, chunk['n']))
        expected_length = self.expected_chunk_length(chunk['n'])
        if len(chunk['data']) != expected_length:
            self.close()
            raise CorruptGridFile('truncated chunk #%d: expected chunk length to be %d but found chunk with length %d' % (chunk['n'], expected_length, len(chunk['data'])))
        self._next_chunk += 1
        return chunk
    __next__ = next

    def close(self) -> None:
        if self._cursor:
            self._cursor.close()
            self._cursor = None