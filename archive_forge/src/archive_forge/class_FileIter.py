import mimetypes
import os
from webob import exc
from webob.dec import wsgify
from webob.response import Response
class FileIter(object):

    def __init__(self, file):
        self.file = file

    def app_iter_range(self, seek=None, limit=None, block_size=None):
        """Iter over the content of the file.

        You can set the `seek` parameter to read the file starting from a
        specific position.

        You can set the `limit` parameter to read the file up to specific
        position.

        Finally, you can change the number of bytes read at once by setting the
        `block_size` parameter.
        """
        if block_size is None:
            block_size = BLOCK_SIZE
        if seek:
            self.file.seek(seek)
            if limit is not None:
                limit -= seek
        try:
            while True:
                data = self.file.read(min(block_size, limit) if limit is not None else block_size)
                if not data:
                    return
                yield data
                if limit is not None:
                    limit -= len(data)
                    if limit <= 0:
                        return
        finally:
            self.file.close()
    __iter__ = app_iter_range