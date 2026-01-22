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
class RevisionInstaller:
    """Installs revisions into a repository"""

    def __init__(self, container, serializer, repository):
        self._container = container
        self._serializer = serializer
        self._repository = repository
        self._info = None

    def install(self):
        """Perform the installation.

        Must be called with the Repository locked.
        """
        with _mod_repository.WriteGroup(self._repository):
            return self._install_in_write_group()

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

    def _handle_info(self, info):
        """Extract data from an info record"""
        self._info = info
        self._source_serializer = self._serializer.get_source_serializer(info)
        if info[b'supports_rich_root'] == 0 and self._repository.supports_rich_root():
            self.update_root = True
        else:
            self.update_root = False

    def _install_mp_records(self, versionedfile, records):
        if len(records) == 0:
            return
        d_func = multiparent.MultiParent.from_patch
        vf_records = [(r, m['parents'], m['sha1'], d_func(t)) for r, m, t in records if r not in versionedfile]
        versionedfile.add_mpdiffs(vf_records)

    def _install_mp_records_keys(self, versionedfile, records):
        d_func = multiparent.MultiParent.from_patch
        vf_records = []
        for key, meta, text in records:
            if len(key) == 2:
                prefix = key[:1]
            else:
                prefix = ()
            parents = [prefix + (parent,) for parent in meta[b'parents']]
            vf_records.append((key, parents, meta[b'sha1'], d_func(text)))
        versionedfile.add_mpdiffs(vf_records)

    def _get_parent_inventory_texts(self, inventory_text_cache, inventory_cache, parent_ids):
        cached_parent_texts = {}
        remaining_parent_ids = []
        for parent_id in parent_ids:
            p_text = inventory_text_cache.get(parent_id, None)
            if p_text is None:
                remaining_parent_ids.append(parent_id)
            else:
                cached_parent_texts[parent_id] = p_text
        ghosts = ()
        if remaining_parent_ids:
            parent_keys = [(r,) for r in remaining_parent_ids]
            present_parent_map = self._repository.inventories.get_parent_map(parent_keys)
            present_parent_ids = []
            ghosts = set()
            for p_id in remaining_parent_ids:
                if (p_id,) in present_parent_map:
                    present_parent_ids.append(p_id)
                else:
                    ghosts.add(p_id)
            to_lines = self._source_serializer.write_inventory_to_chunks
            for parent_inv in self._repository.iter_inventories(present_parent_ids):
                p_text = b''.join(to_lines(parent_inv))
                inventory_cache[parent_inv.revision_id] = parent_inv
                cached_parent_texts[parent_inv.revision_id] = p_text
                inventory_text_cache[parent_inv.revision_id] = p_text
        parent_texts = [cached_parent_texts[parent_id] for parent_id in parent_ids if parent_id not in ghosts]
        return parent_texts

    def _install_inventory_records(self, records):
        if self._info[b'serializer'] == self._repository._serializer.format_num and self._repository._serializer.support_altered_by_hack:
            return self._install_mp_records_keys(self._repository.inventories, records)
        inventory_text_cache = lru_cache.LRUSizeCache(10 * 1024 * 1024)
        inventory_cache = lru_cache.LRUCache(10)
        with ui.ui_factory.nested_progress_bar() as pb:
            num_records = len(records)
            for idx, (key, metadata, bytes) in enumerate(records):
                pb.update('installing inventory', idx, num_records)
                revision_id = key[-1]
                parent_ids = metadata[b'parents']
                p_texts = self._get_parent_inventory_texts(inventory_text_cache, inventory_cache, parent_ids)
                target_lines = multiparent.MultiParent.from_patch(bytes).to_lines(p_texts)
                sha1 = osutils.sha_strings(target_lines)
                if sha1 != metadata[b'sha1']:
                    raise errors.BadBundle("Can't convert to target format")
                inventory_text_cache[revision_id] = b''.join(target_lines)
                target_inv = self._source_serializer.read_inventory_from_lines(target_lines)
                del target_lines
                self._handle_root(target_inv, parent_ids)
                parent_inv = None
                if parent_ids:
                    parent_inv = inventory_cache.get(parent_ids[0], None)
                try:
                    if parent_inv is None:
                        self._repository.add_inventory(revision_id, target_inv, parent_ids)
                    else:
                        delta = target_inv._make_delta(parent_inv)
                        self._repository.add_inventory_by_delta(parent_ids[0], delta, revision_id, parent_ids)
                except serializer.UnsupportedInventoryKind:
                    raise errors.IncompatibleRevision(repr(self._repository))
                inventory_cache[revision_id] = target_inv

    def _handle_root(self, target_inv, parent_ids):
        revision_id = target_inv.revision_id
        if self.update_root:
            text_key = (target_inv.root.file_id, revision_id)
            parent_keys = [(target_inv.root.file_id, parent) for parent in parent_ids]
            self._repository.texts.add_lines(text_key, parent_keys, [])
        elif not self._repository.supports_rich_root():
            if target_inv.root.revision != revision_id:
                raise errors.IncompatibleRevision(repr(self._repository))

    def _install_revision(self, revision_id, metadata, text):
        if self._repository.has_revision(revision_id):
            return
        revision = self._source_serializer.read_revision_from_string(text)
        self._repository.add_revision(revision.revision_id, revision)

    def _install_signature(self, revision_id, metadata, text):
        transaction = self._repository.get_transaction()
        if self._repository.has_signature_for_revision_id(revision_id):
            return
        self._repository.add_signature_text(revision_id, text)