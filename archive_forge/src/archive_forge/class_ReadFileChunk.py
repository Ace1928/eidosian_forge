import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
class ReadFileChunk(object):

    def __init__(self, fileobj, start_byte, chunk_size, full_file_size, callback=None, enable_callback=True):
        """

        Given a file object shown below:

            |___________________________________________________|
            0          |                 |                 full_file_size
                       |----chunk_size---|
                 start_byte

        :type fileobj: file
        :param fileobj: File like object

        :type start_byte: int
        :param start_byte: The first byte from which to start reading.

        :type chunk_size: int
        :param chunk_size: The max chunk size to read.  Trying to read
            pass the end of the chunk size will behave like you've
            reached the end of the file.

        :type full_file_size: int
        :param full_file_size: The entire content length associated
            with ``fileobj``.

        :type callback: function(amount_read)
        :param callback: Called whenever data is read from this object.

        """
        self._fileobj = fileobj
        self._start_byte = start_byte
        self._size = self._calculate_file_size(self._fileobj, requested_size=chunk_size, start_byte=start_byte, actual_file_size=full_file_size)
        self._fileobj.seek(self._start_byte)
        self._amount_read = 0
        self._callback = callback
        self._callback_enabled = enable_callback

    @classmethod
    def from_filename(cls, filename, start_byte, chunk_size, callback=None, enable_callback=True):
        """Convenience factory function to create from a filename.

        :type start_byte: int
        :param start_byte: The first byte from which to start reading.

        :type chunk_size: int
        :param chunk_size: The max chunk size to read.  Trying to read
            pass the end of the chunk size will behave like you've
            reached the end of the file.

        :type full_file_size: int
        :param full_file_size: The entire content length associated
            with ``fileobj``.

        :type callback: function(amount_read)
        :param callback: Called whenever data is read from this object.

        :type enable_callback: bool
        :param enable_callback: Indicate whether to invoke callback
            during read() calls.

        :rtype: ``ReadFileChunk``
        :return: A new instance of ``ReadFileChunk``

        """
        f = open(filename, 'rb')
        file_size = os.fstat(f.fileno()).st_size
        return cls(f, start_byte, chunk_size, file_size, callback, enable_callback)

    def _calculate_file_size(self, fileobj, requested_size, start_byte, actual_file_size):
        max_chunk_size = actual_file_size - start_byte
        return min(max_chunk_size, requested_size)

    def read(self, amount=None):
        if amount is None:
            amount_to_read = self._size - self._amount_read
        else:
            amount_to_read = min(self._size - self._amount_read, amount)
        data = self._fileobj.read(amount_to_read)
        self._amount_read += len(data)
        if self._callback is not None and self._callback_enabled:
            self._callback(len(data))
        return data

    def enable_callback(self):
        self._callback_enabled = True

    def disable_callback(self):
        self._callback_enabled = False

    def seek(self, where):
        self._fileobj.seek(self._start_byte + where)
        if self._callback is not None and self._callback_enabled:
            self._callback(where - self._amount_read)
        self._amount_read = where

    def close(self):
        self._fileobj.close()

    def tell(self):
        return self._amount_read

    def __len__(self):
        return self._size

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __iter__(self):
        return iter([])