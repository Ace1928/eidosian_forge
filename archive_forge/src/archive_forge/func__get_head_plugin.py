from bz2 import BZ2File
from gzip import GzipFile
from io import IOBase
from lzma import LZMAFile
from typing import Any, BinaryIO, Dict, List
from scrapy.utils.misc import load_object
def _get_head_plugin(self) -> Any:
    prev = self.file
    for plugin in self.plugins[::-1]:
        prev = plugin(prev, self.feed_options)
    return prev