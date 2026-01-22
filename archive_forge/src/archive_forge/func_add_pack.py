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
def add_pack(self):
    """Add a new pack to this object store.

        Returns: Fileobject to write to and a commit function to
            call when the pack is finished.
        """
    f = BytesIO()

    def commit():
        f.seek(0)
        pack = PackData(file=f, filename='')
        entries = pack.sorted_entries()
        if entries:
            basename = posixpath.join(self.pack_dir, 'pack-%s' % iter_sha1((entry[0] for entry in entries)))
            index = BytesIO()
            write_pack_index_v2(index, entries, pack.get_stored_checksum())
            self.scon.put_object(basename + '.pack', f)
            f.close()
            self.scon.put_object(basename + '.idx', index)
            index.close()
            final_pack = SwiftPack(basename, scon=self.scon)
            final_pack.check_length_and_checksum()
            self._add_cached_pack(basename, final_pack)
            return final_pack
        else:
            return None

    def abort():
        pass
    return (f, commit, abort)