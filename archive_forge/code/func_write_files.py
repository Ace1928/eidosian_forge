import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def write_files(self):
    """Write bundle records for all revisions of all files"""
    text_keys = []
    altered_fileids = self.repository.fileids_altered_by_revision_ids(self.revision_ids)
    for file_id, revision_ids in altered_fileids.items():
        for revision_id in revision_ids:
            text_keys.append((file_id, revision_id))
    self._add_mp_records_keys('file', self.repository.texts, text_keys)