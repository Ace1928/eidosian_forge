import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
class BucketBasedObjectStore(PackBasedObjectStore):
    """Object store implementation that uses a bucket store like S3 as backend."""

    def _iter_loose_objects(self):
        """Iterate over the SHAs of all loose objects."""
        return iter([])

    def _get_loose_object(self, sha):
        return None

    def _remove_loose_object(self, sha):
        pass

    def _remove_pack(self, name):
        raise NotImplementedError(self._remove_pack)

    def _iter_pack_names(self):
        raise NotImplementedError(self._iter_pack_names)

    def _get_pack(self, name):
        raise NotImplementedError(self._get_pack)

    def _update_pack_cache(self):
        pack_files = set(self._iter_pack_names())
        new_packs = []
        for f in pack_files:
            if f not in self._pack_cache:
                pack = self._get_pack(f)
                new_packs.append(pack)
                self._pack_cache[f] = pack
        for f in set(self._pack_cache) - pack_files:
            self._pack_cache.pop(f).close()
        return new_packs

    def _upload_pack(self, basename, pack_file, index_file):
        raise NotImplementedError

    def add_pack(self):
        """Add a new pack to this object store.

        Returns: Fileobject to write to, a commit function to
            call when the pack is finished and an abort
            function.
        """
        import tempfile
        pf = tempfile.SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix='incoming-')

        def commit():
            if pf.tell() == 0:
                pf.close()
                return None
            pf.seek(0)
            p = PackData(pf.name, pf)
            entries = p.sorted_entries()
            basename = iter_sha1((entry[0] for entry in entries)).decode('ascii')
            idxf = tempfile.SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix='incoming-')
            checksum = p.get_stored_checksum()
            write_pack_index(idxf, entries, checksum)
            idxf.seek(0)
            idx = load_pack_index_file(basename + '.idx', idxf)
            for pack in self.packs:
                if pack.get_stored_checksum() == p.get_stored_checksum():
                    p.close()
                    idx.close()
                    return pack
            pf.seek(0)
            idxf.seek(0)
            self._upload_pack(basename, pf, idxf)
            final_pack = Pack.from_objects(p, idx)
            self._add_cached_pack(basename, final_pack)
            return final_pack
        return (pf, commit, pf.close)