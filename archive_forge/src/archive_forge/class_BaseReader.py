import re
from io import BytesIO
from .. import errors
class BaseReader:

    def __init__(self, source_file):
        """Constructor.

        :param source_file: a file-like object with `read` and `readline`
            methods.
        """
        self._source = source_file

    def reader_func(self, length=None):
        return self._source.read(length)

    def _read_line(self):
        line = self._source.readline()
        if not line.endswith(b'\n'):
            raise UnexpectedEndOfContainerError()
        return line.rstrip(b'\n')