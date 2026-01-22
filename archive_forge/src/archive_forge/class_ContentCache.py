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
class ContentCache:
    """Object that can cache Git objects."""

    def add(self, object):
        """Add an object."""
        raise NotImplementedError(self.add)

    def add_multi(self, objects):
        """Add multiple objects."""
        for obj in objects:
            self.add(obj)

    def __getitem__(self, sha):
        """Retrieve an item, by SHA."""
        raise NotImplementedError(self.__getitem__)