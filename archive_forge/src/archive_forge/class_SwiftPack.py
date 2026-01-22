import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
class SwiftPack(Pack):
    """A Git pack object.

    Same implementation as pack.Pack except that _idx_load and
    _data_load are bounded to Swift version of load_pack_index and
    PackData.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.scon = kwargs['scon']
        del kwargs['scon']
        super().__init__(*args, **kwargs)
        self._pack_info_path = self._basename + '.info'
        self._pack_info = None
        self._pack_info_load = lambda: load_pack_info(self._pack_info_path, self.scon)
        self._idx_load = lambda: swift_load_pack_index(self.scon, self._idx_path)
        self._data_load = lambda: SwiftPackData(self.scon, self._data_path)

    @property
    def pack_info(self):
        """The pack data object being used."""
        if self._pack_info is None:
            self._pack_info = self._pack_info_load()
        return self._pack_info