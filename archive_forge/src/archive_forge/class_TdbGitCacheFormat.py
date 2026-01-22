import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
class TdbGitCacheFormat(BzrGitCacheFormat):
    """Cache format for tdb-based caches."""

    def get_format_string(self):
        return b'bzr-git sha map version 3 using tdb\n'

    def open(self, transport):
        try:
            basepath = transport.local_abspath('.')
        except bzr_errors.NotLocalUrl:
            basepath = get_cache_dir()
        try:
            return TdbBzrGitCache(os.path.join(basepath, 'idmap.tdb'))
        except ImportError:
            raise ImportError("Unable to open existing bzr-git cache because 'tdb' is not installed.")