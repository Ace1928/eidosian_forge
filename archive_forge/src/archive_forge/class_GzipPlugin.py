from bz2 import BZ2File
from gzip import GzipFile
from io import IOBase
from lzma import LZMAFile
from typing import Any, BinaryIO, Dict, List
from scrapy.utils.misc import load_object
class GzipPlugin:
    """
    Compresses received data using `gzip <https://en.wikipedia.org/wiki/Gzip>`_.

    Accepted ``feed_options`` parameters:

    - `gzip_compresslevel`
    - `gzip_mtime`
    - `gzip_filename`

    See :py:class:`gzip.GzipFile` for more info about parameters.
    """

    def __init__(self, file: BinaryIO, feed_options: Dict[str, Any]) -> None:
        self.file = file
        self.feed_options = feed_options
        compress_level = self.feed_options.get('gzip_compresslevel', 9)
        mtime = self.feed_options.get('gzip_mtime')
        filename = self.feed_options.get('gzip_filename')
        self.gzipfile = GzipFile(fileobj=self.file, mode='wb', compresslevel=compress_level, mtime=mtime, filename=filename)

    def write(self, data: bytes) -> int:
        return self.gzipfile.write(data)

    def close(self) -> None:
        self.gzipfile.close()
        self.file.close()