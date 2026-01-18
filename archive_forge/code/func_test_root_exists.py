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
def test_root_exists(self):
    """Check that Swift container exist.

        Returns: True if exist or None it not
        """
    ret = self.httpclient.request('HEAD', self.base_path)
    if ret.status_code == 404:
        return None
    if ret.status_code < 200 or ret.status_code > 300:
        raise SwiftException('HEAD request failed with error code %s' % ret.status_code)
    return True