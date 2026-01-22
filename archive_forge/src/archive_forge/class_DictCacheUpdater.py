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
class DictCacheUpdater(CacheUpdater):
    """Cache updater for dict-based caches."""

    def __init__(self, cache, rev):
        self.cache = cache
        self.revid = rev.revision_id
        self.parent_revids = rev.parent_ids
        self._commit = None
        self._entries = []

    def add_object(self, obj, bzr_key_data, path):
        if isinstance(obj, tuple):
            type_name, hexsha = obj
        else:
            type_name = obj.type_name.decode('ascii')
            hexsha = obj.id
        if not isinstance(hexsha, bytes):
            raise TypeError(hexsha)
        if type_name == 'commit':
            self._commit = obj
            if type(bzr_key_data) is not dict:
                raise TypeError(bzr_key_data)
            key = self.revid
            type_data = (self.revid, self._commit.tree, bzr_key_data)
            self.cache.idmap._by_revid[self.revid] = hexsha
        elif type_name in ('blob', 'tree'):
            if bzr_key_data is not None:
                key = type_data = bzr_key_data
                self.cache.idmap._by_fileid.setdefault(type_data[1], {})[type_data[0]] = hexsha
        else:
            raise AssertionError
        entry = (type_name, type_data)
        self.cache.idmap._by_sha.setdefault(hexsha, {})[key] = entry

    def finish(self):
        if self._commit is None:
            raise AssertionError('No commit object added')
        return self._commit