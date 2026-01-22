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
def _install_in_write_group(self):
    current_file = None
    current_versionedfile = None
    pending_file_records = []
    inventory_vf = None
    pending_inventory_records = []
    added_inv = set()
    target_revision = None
    for bytes, metadata, repo_kind, revision_id, file_id in self._container.iter_records():
        if repo_kind == 'info':
            if self._info is not None:
                raise AssertionError()
            self._handle_info(metadata)
        if pending_file_records and (repo_kind, file_id) != ('file', current_file):
            self._install_mp_records_keys(self._repository.texts, pending_file_records)
            current_file = None
            del pending_file_records[:]
        if len(pending_inventory_records) > 0 and repo_kind != 'inventory':
            self._install_inventory_records(pending_inventory_records)
            pending_inventory_records = []
        if repo_kind == 'inventory':
            pending_inventory_records.append(((revision_id,), metadata, bytes))
        if repo_kind == 'revision':
            target_revision = revision_id
            self._install_revision(revision_id, metadata, bytes)
        if repo_kind == 'signature':
            self._install_signature(revision_id, metadata, bytes)
        if repo_kind == 'file':
            current_file = file_id
            pending_file_records.append(((file_id, revision_id), metadata, bytes))
    self._install_mp_records_keys(self._repository.texts, pending_file_records)
    return target_revision