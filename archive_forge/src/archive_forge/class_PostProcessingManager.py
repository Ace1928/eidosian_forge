from bz2 import BZ2File
from gzip import GzipFile
from io import IOBase
from lzma import LZMAFile
from typing import Any, BinaryIO, Dict, List
from scrapy.utils.misc import load_object
class PostProcessingManager(IOBase):
    """
    This will manage and use declared plugins to process data in a
    pipeline-ish way.
    :param plugins: all the declared plugins for the feed
    :type plugins: list
    :param file: final target file where the processed data will be written
    :type file: file like object
    """

    def __init__(self, plugins: List[Any], file: BinaryIO, feed_options: Dict[str, Any]) -> None:
        self.plugins = self._load_plugins(plugins)
        self.file = file
        self.feed_options = feed_options
        self.head_plugin = self._get_head_plugin()

    def write(self, data: bytes) -> int:
        """
        Uses all the declared plugins to process data first, then writes
        the processed data to target file.
        :param data: data passed to be written to target file
        :type data: bytes
        :return: returns number of bytes written
        :rtype: int
        """
        return self.head_plugin.write(data)

    def tell(self) -> int:
        return self.file.tell()

    def close(self) -> None:
        """
        Close the target file along with all the plugins.
        """
        self.head_plugin.close()

    def writable(self) -> bool:
        return True

    def _load_plugins(self, plugins: List[Any]) -> List[Any]:
        plugins = [load_object(plugin) for plugin in plugins]
        return plugins

    def _get_head_plugin(self) -> Any:
        prev = self.file
        for plugin in self.plugins[::-1]:
            prev = plugin(prev, self.feed_options)
        return prev