import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
class ReadableToIterable:
    """
    Wrap a filelike object and act as an iterator.

    It is recommended to use this class only on files opened in binary mode.
    Due to the Unicode changes in Python 3, files are now opened using an
    encoding not suitable for use with the md5 class and because of this
    hit the exception on every call to next. This could cause problems,
    especially with large files and small chunk sizes.
    """

    def __init__(self, content, chunk_size=65536, md5=False):
        """
        :param content: The filelike object that is yielded from.
        :param chunk_size: The max size of each yielded item.
        :param md5: Flag to enable calculating the MD5 of the content
                    as it is yielded.
        """
        self.md5sum = hashlib.md5() if md5 else NoopMD5()
        self.content = content
        self.chunk_size = chunk_size

    def get_md5sum(self):
        return self.md5sum.hexdigest()

    def __next__(self):
        """
        Both ``__next__`` and ``next`` are provided to allow compatibility
        with python 2 and python 3 and their use of ``iterable.next()``
        and ``next(iterable)`` respectively.
        """
        chunk = self.content.read(self.chunk_size)
        if not chunk:
            raise StopIteration
        try:
            self.md5sum.update(chunk)
        except TypeError:
            self.md5sum.update(chunk.encode())
        return chunk

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self