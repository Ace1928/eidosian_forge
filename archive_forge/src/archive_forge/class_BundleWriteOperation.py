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
class BundleWriteOperation:
    """Perform the operation of writing revisions to a bundle"""

    def __init__(self, base, target, repository, fileobj, revision_ids=None):
        self.base = base
        self.target = target
        self.repository = repository
        bundle = BundleWriter(fileobj)
        self.bundle = bundle
        if revision_ids is not None:
            self.revision_ids = revision_ids
        else:
            graph = repository.get_graph()
            revision_ids = graph.find_unique_ancestors(target, [base])
            parents = graph.get_parent_map(revision_ids)
            self.revision_ids = [r for r in revision_ids if r in parents]
        self.revision_keys = {(revid,) for revid in self.revision_ids}

    def do_write(self):
        """Write all data to the bundle"""
        trace.note(ngettext('Bundling %d revision.', 'Bundling %d revisions.', len(self.revision_ids)), len(self.revision_ids))
        with self.repository.lock_read():
            self.bundle.begin()
            self.write_info()
            self.write_files()
            self.write_revisions()
            self.bundle.end()
        return self.revision_ids

    def write_info(self):
        """Write format info"""
        serializer_format = self.repository.get_serializer_format()
        supports_rich_root = {True: 1, False: 0}[self.repository.supports_rich_root()]
        self.bundle.add_info_record({b'serializer': serializer_format, b'supports_rich_root': supports_rich_root})

    def write_files(self):
        """Write bundle records for all revisions of all files"""
        text_keys = []
        altered_fileids = self.repository.fileids_altered_by_revision_ids(self.revision_ids)
        for file_id, revision_ids in altered_fileids.items():
            for revision_id in revision_ids:
                text_keys.append((file_id, revision_id))
        self._add_mp_records_keys('file', self.repository.texts, text_keys)

    def write_revisions(self):
        """Write bundle records for all revisions and signatures"""
        inv_vf = self.repository.inventories
        topological_order = [key[-1] for key in multiparent.topo_iter_keys(inv_vf, self.revision_keys)]
        revision_order = topological_order
        if self.target is not None and self.target in self.revision_ids:
            revision_order = list(topological_order)
            revision_order.remove(self.target)
            revision_order.append(self.target)
        if self.repository._serializer.support_altered_by_hack:
            self._add_mp_records_keys('inventory', inv_vf, [(revid,) for revid in topological_order])
        else:
            self._add_inventory_mpdiffs_from_serializer(topological_order)
        self._add_revision_texts(revision_order)

    def _add_inventory_mpdiffs_from_serializer(self, revision_order):
        """Generate mpdiffs by serializing inventories.

        The current repository only has part of the tree shape information in
        the 'inventories' vf. So we use serializer.write_inventory_to_lines to
        get a 'full' representation of the tree shape, and then generate
        mpdiffs on that data stream. This stream can then be reconstructed on
        the other side.
        """
        inventory_key_order = [(r,) for r in revision_order]
        generator = _MPDiffInventoryGenerator(self.repository, inventory_key_order)
        for revision_id, parent_ids, sha1, diff in generator.iter_diffs():
            text = b''.join(diff.to_patch())
            self.bundle.add_multiparent_record(text, sha1, parent_ids, 'inventory', revision_id, None)

    def _add_revision_texts(self, revision_order):
        parent_map = self.repository.get_parent_map(revision_order)
        revision_to_bytes = self.repository._serializer.write_revision_to_string
        revisions = self.repository.get_revisions(revision_order)
        for revision in revisions:
            revision_id = revision.revision_id
            parents = parent_map.get(revision_id, None)
            revision_text = revision_to_bytes(revision)
            self.bundle.add_fulltext_record(revision_text, parents, 'revision', revision_id)
            try:
                self.bundle.add_fulltext_record(self.repository.get_signature_text(revision_id), parents, 'signature', revision_id)
            except errors.NoSuchRevision:
                pass

    @staticmethod
    def get_base_target(revision_ids, forced_bases, repository):
        """Determine the base and target from old-style revision ids"""
        if len(revision_ids) == 0:
            return (None, None)
        target = revision_ids[0]
        base = forced_bases.get(target)
        if base is None:
            parents = repository.get_revision(target).parent_ids
            if len(parents) == 0:
                base = _mod_revision.NULL_REVISION
            else:
                base = parents[0]
        return (base, target)

    def _add_mp_records_keys(self, repo_kind, vf, keys):
        """Add multi-parent diff records to a bundle"""
        ordered_keys = list(multiparent.topo_iter_keys(vf, keys))
        mpdiffs = vf.make_mpdiffs(ordered_keys)
        sha1s = vf.get_sha1s(ordered_keys)
        parent_map = vf.get_parent_map(ordered_keys)
        for mpdiff, item_key in zip(mpdiffs, ordered_keys):
            sha1 = sha1s[item_key]
            parents = [key[-1] for key in parent_map[item_key]]
            text = b''.join(mpdiff.to_patch())
            if len(item_key) == 2:
                file_id = item_key[0]
            else:
                file_id = None
            self.bundle.add_multiparent_record(text, sha1, parents, repo_kind, item_key[-1], file_id)