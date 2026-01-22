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
class VersionedFilesContentCache(ContentCache):

    def __init__(self, vf):
        self._vf = vf

    def add(self, obj):
        self._vf.insert_record_stream([versionedfile.ChunkedContentFactory((obj.id,), [], None, obj.as_legacy_object_chunks())])

    def __getitem__(self, sha):
        stream = self._vf.get_record_stream([(sha,)], 'unordered', True)
        entry = next(stream)
        if entry.storage_kind == 'absent':
            raise KeyError(sha)
        return ShaFile._parse_legacy_object(entry.get_bytes_as('fulltext'))