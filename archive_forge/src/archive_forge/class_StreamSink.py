from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
class StreamSink:
    """An object that can insert a stream into a repository.

    This interface handles the complexity of reserialising inventories and
    revisions from different formats, and allows unidirectional insertion into
    stacked repositories without looking for the missing basis parents
    beforehand.
    """

    def __init__(self, target_repo):
        self.target_repo = target_repo

    def insert_missing_keys(self, source, missing_keys):
        """Insert missing keys from another source.

        :param source: StreamSource to stream from
        :param missing_keys: Keys to insert
        :return: keys still missing
        """
        stream = source.get_stream_for_missing_keys(missing_keys)
        return self.insert_stream_without_locking(stream, self.target_repo._format)

    def insert_stream(self, stream, src_format, resume_tokens):
        """Insert a stream's content into the target repository.

        :param src_format: a bzr repository format.

        :return: a list of resume tokens and an  iterable of keys additional
            items required before the insertion can be completed.
        """
        with self.target_repo.lock_write():
            if resume_tokens:
                self.target_repo.resume_write_group(resume_tokens)
                is_resume = True
            else:
                self.target_repo.start_write_group()
                is_resume = False
            try:
                missing_keys = self.insert_stream_without_locking(stream, src_format, is_resume)
                if missing_keys:
                    write_group_tokens = self.target_repo.suspend_write_group()
                    return (write_group_tokens, missing_keys)
                hint = self.target_repo.commit_write_group()
                to_serializer = self.target_repo._format._serializer
                src_serializer = src_format._serializer
                if to_serializer != src_serializer and self.target_repo._format.pack_compresses:
                    self.target_repo.pack(hint=hint)
                return ([], set())
            except:
                self.target_repo.abort_write_group(suppress_errors=True)
                raise

    def insert_stream_without_locking(self, stream, src_format, is_resume=False):
        """Insert a stream's content into the target repository.

        This assumes that you already have a locked repository and an active
        write group.

        :param src_format: a bzr repository format.
        :param is_resume: Passed down to get_missing_parent_inventories to
            indicate if we should be checking for missing texts at the same
            time.

        :return: A set of keys that are missing.
        """
        if not self.target_repo.is_write_locked():
            raise errors.ObjectNotLocked(self)
        if not self.target_repo.is_in_write_group():
            raise errors.BzrError('you must already be in a write group')
        to_serializer = self.target_repo._format._serializer
        src_serializer = src_format._serializer
        new_pack = None
        if to_serializer == src_serializer:
            try:
                new_pack = self.target_repo._pack_collection._new_pack
            except AttributeError:
                pass
            else:
                new_pack.set_write_cache_size(1024 * 1024)
        for substream_type, substream in stream:
            if 'stream' in debug.debug_flags:
                mutter('inserting substream: %s', substream_type)
            if substream_type == 'texts':
                self.target_repo.texts.insert_record_stream(substream)
            elif substream_type == 'inventories':
                if src_serializer == to_serializer:
                    self.target_repo.inventories.insert_record_stream(substream)
                else:
                    self._extract_and_insert_inventories(substream, src_serializer)
            elif substream_type == 'inventory-deltas':
                self._extract_and_insert_inventory_deltas(substream, src_serializer)
            elif substream_type == 'chk_bytes':
                self.target_repo.chk_bytes.insert_record_stream(substream)
            elif substream_type == 'revisions':
                if src_serializer == to_serializer:
                    self.target_repo.revisions.insert_record_stream(substream)
                else:
                    self._extract_and_insert_revisions(substream, src_serializer)
            elif substream_type == 'signatures':
                self.target_repo.signatures.insert_record_stream(substream)
            else:
                raise AssertionError('kaboom! {}'.format(substream_type))
        if new_pack is not None:
            new_pack._write_data(b'', flush=True)
        missing_keys = self.target_repo.get_missing_parent_inventories(check_for_missing_texts=is_resume)
        try:
            for prefix, versioned_file in (('texts', self.target_repo.texts), ('inventories', self.target_repo.inventories), ('revisions', self.target_repo.revisions), ('signatures', self.target_repo.signatures), ('chk_bytes', self.target_repo.chk_bytes)):
                if versioned_file is None:
                    continue
                missing_keys.update(((prefix,) + key for key in versioned_file.get_missing_compression_parent_keys()))
        except NotImplementedError:
            missing_keys = set()
        return missing_keys

    def _extract_and_insert_inventory_deltas(self, substream, serializer):
        target_rich_root = self.target_repo._format.rich_root_data
        target_tree_refs = self.target_repo._format.supports_tree_reference
        for record in substream:
            inventory_delta_bytes = record.get_bytes_as('lines')
            deserialiser = inventory_delta.InventoryDeltaDeserializer()
            try:
                parse_result = deserialiser.parse_text_bytes(inventory_delta_bytes)
            except inventory_delta.IncompatibleInventoryDelta as err:
                mutter('Incompatible delta: %s', err.msg)
                raise errors.IncompatibleRevision(self.target_repo._format)
            basis_id, new_id, rich_root, tree_refs, inv_delta = parse_result
            revision_id = new_id
            parents = [key[0] for key in record.parents]
            self.target_repo.add_inventory_by_delta(basis_id, inv_delta, revision_id, parents)

    def _extract_and_insert_inventories(self, substream, serializer, parse_delta=None):
        """Generate a new inventory versionedfile in target, converting data.

        The inventory is retrieved from the source, (deserializing it), and
        stored in the target (reserializing it in a different format).
        """
        target_rich_root = self.target_repo._format.rich_root_data
        target_tree_refs = self.target_repo._format.supports_tree_reference
        for record in substream:
            lines = record.get_bytes_as('lines')
            revision_id = record.key[0]
            inv = serializer.read_inventory_from_lines(lines, revision_id)
            parents = [key[0] for key in record.parents]
            self.target_repo.add_inventory(revision_id, inv, parents)
            del inv

    def _extract_and_insert_revisions(self, substream, serializer):
        for record in substream:
            bytes = record.get_bytes_as('fulltext')
            revision_id = record.key[0]
            rev = serializer.read_revision_from_string(bytes)
            if rev.revision_id != revision_id:
                raise AssertionError('wtf: {} != {}'.format(rev, revision_id))
            self.target_repo.add_revision(revision_id, rev)

    def finished(self):
        if self.target_repo._format._fetch_reconcile:
            self.target_repo.reconcile()