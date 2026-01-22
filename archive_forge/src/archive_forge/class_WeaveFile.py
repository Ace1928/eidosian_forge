import os
from copy import copy
from io import BytesIO
import patiencediff
from ..lazy_import import lazy_import
from breezy import tsort
from .. import errors, osutils
from .. import transport as _mod_transport
from ..errors import RevisionAlreadyPresent, RevisionNotPresent
from ..osutils import dirname, sha, sha_strings, split_lines
from ..revision import NULL_REVISION
from ..trace import mutter
from .versionedfile import (AbsentContentFactory, ContentFactory,
from .weavefile import _read_weave_v5, write_weave_v5
class WeaveFile(Weave):
    """A WeaveFile represents a Weave on disk and writes on change."""
    WEAVE_SUFFIX = '.weave'

    def __init__(self, name, transport, filemode=None, create=False, access_mode='w', get_scope=None):
        """Create a WeaveFile.

        :param create: If not True, only open an existing knit.
        """
        super().__init__(name, access_mode, get_scope=get_scope, allow_reserved=False)
        self._transport = transport
        self._filemode = filemode
        try:
            with self._transport.get(name + WeaveFile.WEAVE_SUFFIX) as f:
                _read_weave_v5(BytesIO(f.read()), self)
        except _mod_transport.NoSuchFile:
            if not create:
                raise
            self._save()

    def _add_lines(self, version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content):
        """Add a version and save the weave."""
        self.check_not_reserved_id(version_id)
        result = super()._add_lines(version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content)
        self._save()
        return result

    def copy_to(self, name, transport):
        """See VersionedFile.copy_to()."""
        sio = BytesIO()
        write_weave_v5(self, sio)
        sio.seek(0)
        transport.put_file(name + WeaveFile.WEAVE_SUFFIX, sio, self._filemode)

    def _save(self):
        """Save the weave."""
        self._check_write_ok()
        sio = BytesIO()
        write_weave_v5(self, sio)
        sio.seek(0)
        bytes = sio.getvalue()
        path = self._weave_name + WeaveFile.WEAVE_SUFFIX
        try:
            self._transport.put_bytes(path, bytes, self._filemode)
        except _mod_transport.NoSuchFile:
            self._transport.mkdir(dirname(path))
            self._transport.put_bytes(path, bytes, self._filemode)

    @staticmethod
    def get_suffixes():
        """See VersionedFile.get_suffixes()."""
        return [WeaveFile.WEAVE_SUFFIX]

    def insert_record_stream(self, stream):
        super().insert_record_stream(stream)
        self._save()