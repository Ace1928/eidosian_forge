from bz2 import BZ2File
from gzip import GzipFile
from io import IOBase
from lzma import LZMAFile
from typing import Any, BinaryIO, Dict, List
from scrapy.utils.misc import load_object
class Bz2Plugin:
    """
    Compresses received data using `bz2 <https://en.wikipedia.org/wiki/Bzip2>`_.

    Accepted ``feed_options`` parameters:

    - `bz2_compresslevel`

    See :py:class:`bz2.BZ2File` for more info about parameters.
    """

    def __init__(self, file: BinaryIO, feed_options: Dict[str, Any]) -> None:
        self.file = file
        self.feed_options = feed_options
        compress_level = self.feed_options.get('bz2_compresslevel', 9)
        self.bz2file = BZ2File(filename=self.file, mode='wb', compresslevel=compress_level)

    def write(self, data: bytes) -> int:
        return self.bz2file.write(data)

    def close(self) -> None:
        self.bz2file.close()
        self.file.close()