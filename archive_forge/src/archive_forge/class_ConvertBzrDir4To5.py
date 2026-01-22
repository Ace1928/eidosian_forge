from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
class ConvertBzrDir4To5(Converter):
    """Converts format 4 bzr dirs to format 5."""

    def __init__(self):
        super().__init__()
        self.converted_revs = set()
        self.absent_revisions = set()
        self.text_count = 0
        self.revisions = {}

    def convert(self, to_convert, pb):
        """See Converter.convert()."""
        self.controldir = to_convert
        with ui.ui_factory.nested_progress_bar() as self.pb:
            ui.ui_factory.note(gettext('starting upgrade from format 4 to 5'))
            if isinstance(self.controldir.transport, local.LocalTransport):
                self.controldir.get_workingtree_transport(None).delete('stat-cache')
            self._convert_to_weaves()
            return ControlDir.open(self.controldir.user_url)

    def _convert_to_weaves(self):
        ui.ui_factory.note(gettext('note: upgrade may be faster if all store files are ungzipped first'))
        try:
            stat = self.controldir.transport.stat('weaves')
            if not S_ISDIR(stat.st_mode):
                self.controldir.transport.delete('weaves')
                self.controldir.transport.mkdir('weaves')
        except NoSuchFile:
            self.controldir.transport.mkdir('weaves')
        self.inv_weave = weave.Weave('inventory')
        self.text_weaves = {}
        self.controldir.transport.delete('branch-format')
        self.branch = self.controldir.open_branch()
        self._convert_working_inv()
        rev_history = self.branch._revision_history()
        self.known_revisions = set(rev_history)
        self.to_read = rev_history[-1:]
        while self.to_read:
            rev_id = self.to_read.pop()
            if rev_id not in self.revisions and rev_id not in self.absent_revisions:
                self._load_one_rev(rev_id)
        self.pb.clear()
        to_import = self._make_order()
        for i, rev_id in enumerate(to_import):
            self.pb.update(gettext('converting revision'), i, len(to_import))
            self._convert_one_rev(rev_id)
        self.pb.clear()
        self._write_all_weaves()
        self._write_all_revs()
        ui.ui_factory.note(gettext('upgraded to weaves:'))
        ui.ui_factory.note('  ' + gettext('%6d revisions and inventories') % len(self.revisions))
        ui.ui_factory.note('  ' + gettext('%6d revisions not present') % len(self.absent_revisions))
        ui.ui_factory.note('  ' + gettext('%6d texts') % self.text_count)
        self._cleanup_spare_files_after_format4()
        self.branch._transport.put_bytes('branch-format', BzrDirFormat5().get_format_string(), mode=self.controldir._get_file_mode())

    def _cleanup_spare_files_after_format4(self):
        for n in ('merged-patches', 'pending-merged-patches'):
            try:
                self.controldir.transport.delete(n)
            except NoSuchFile:
                pass
        self.controldir.transport.delete_tree('inventory-store')
        self.controldir.transport.delete_tree('text-store')

    def _convert_working_inv(self):
        inv = xml4.serializer_v4.read_inventory(self.branch._transport.get('inventory'))
        f = BytesIO()
        xml5.serializer_v5.write_inventory(inv, f, working=True)
        self.branch._transport.put_bytes('inventory', f.getvalue(), mode=self.controldir._get_file_mode())

    def _write_all_weaves(self):
        controlweaves = VersionedFileStore(self.controldir.transport, prefixed=False, versionedfile_class=weave.WeaveFile)
        weave_transport = self.controldir.transport.clone('weaves')
        weaves = VersionedFileStore(weave_transport, prefixed=False, versionedfile_class=weave.WeaveFile)
        transaction = WriteTransaction()
        try:
            i = 0
            for file_id, file_weave in self.text_weaves.items():
                self.pb.update(gettext('writing weave'), i, len(self.text_weaves))
                weaves._put_weave(file_id, file_weave, transaction)
                i += 1
            self.pb.update(gettext('inventory'), 0, 1)
            controlweaves._put_weave(b'inventory', self.inv_weave, transaction)
            self.pb.update(gettext('inventory'), 1, 1)
        finally:
            self.pb.clear()

    def _write_all_revs(self):
        """Write all revisions out in new form."""
        self.controldir.transport.delete_tree('revision-store')
        self.controldir.transport.mkdir('revision-store')
        revision_transport = self.controldir.transport.clone('revision-store')
        from ...bzr.xml5 import serializer_v5
        from .repository import RevisionTextStore
        revision_store = RevisionTextStore(revision_transport, serializer_v5, False, versionedfile.PrefixMapper(), lambda: True, lambda: True)
        try:
            for i, rev_id in enumerate(self.converted_revs):
                self.pb.update(gettext('write revision'), i, len(self.converted_revs))
                lines = serializer_v5.write_revision_to_lines(self.revisions[rev_id])
                key = (rev_id,)
                revision_store.add_lines(key, None, lines)
        finally:
            self.pb.clear()

    def _load_one_rev(self, rev_id):
        """Load a revision object into memory.

        Any parents not either loaded or abandoned get queued to be
        loaded."""
        self.pb.update(gettext('loading revision'), len(self.revisions), len(self.known_revisions))
        if not self.branch.repository.has_revision(rev_id):
            self.pb.clear()
            ui.ui_factory.note(gettext('revision {%s} not present in branch; will be converted as a ghost') % rev_id)
            self.absent_revisions.add(rev_id)
        else:
            rev = self.branch.repository.get_revision(rev_id)
            for parent_id in rev.parent_ids:
                self.known_revisions.add(parent_id)
                self.to_read.append(parent_id)
            self.revisions[rev_id] = rev

    def _load_old_inventory(self, rev_id):
        with self.branch.repository.inventory_store.get(rev_id) as f:
            inv = xml4.serializer_v4.read_inventory(f)
        inv.revision_id = rev_id
        rev = self.revisions[rev_id]
        return inv

    def _load_updated_inventory(self, rev_id):
        inv_xml = self.inv_weave.get_lines(rev_id)
        inv = xml5.serializer_v5.read_inventory_from_lines(inv_xml, rev_id)
        return inv

    def _convert_one_rev(self, rev_id):
        """Convert revision and all referenced objects to new format."""
        rev = self.revisions[rev_id]
        inv = self._load_old_inventory(rev_id)
        present_parents = [p for p in rev.parent_ids if p not in self.absent_revisions]
        self._convert_revision_contents(rev, inv, present_parents)
        self._store_new_inv(rev, inv, present_parents)
        self.converted_revs.add(rev_id)

    def _store_new_inv(self, rev, inv, present_parents):
        new_inv_xml = xml5.serializer_v5.write_inventory_to_lines(inv)
        new_inv_sha1 = osutils.sha_strings(new_inv_xml)
        self.inv_weave.add_lines(rev.revision_id, present_parents, new_inv_xml)
        rev.inventory_sha1 = new_inv_sha1

    def _convert_revision_contents(self, rev, inv, present_parents):
        """Convert all the files within a revision.

        Also upgrade the inventory to refer to the text revision ids."""
        rev_id = rev.revision_id
        trace.mutter('converting texts of revision {%s}', rev_id)
        parent_invs = list(map(self._load_updated_inventory, present_parents))
        entries = inv.iter_entries()
        next(entries)
        for path, ie in entries:
            self._convert_file_version(rev, ie, parent_invs)

    def _convert_file_version(self, rev, ie, parent_invs):
        """Convert one version of one file.

        The file needs to be added into the weave if it is a merge
        of >=2 parents or if it's changed from its parent.
        """
        file_id = ie.file_id
        rev_id = rev.revision_id
        w = self.text_weaves.get(file_id)
        if w is None:
            w = weave.Weave(file_id)
            self.text_weaves[file_id] = w
        text_changed = False
        parent_candiate_entries = ie.parent_candidates(parent_invs)
        heads = graph.Graph(self).heads(parent_candiate_entries)
        previous_entries = {head: parent_candiate_entries[head] for head in heads}
        self.snapshot_ie(previous_entries, ie, w, rev_id)

    def get_parent_map(self, revision_ids):
        """See graph.StackedParentsProvider.get_parent_map"""
        return {revision_id: self.revisions[revision_id] for revision_id in revision_ids if revision_id in self.revisions}

    def snapshot_ie(self, previous_revisions, ie, w, rev_id):
        if len(previous_revisions) == 1:
            previous_ie = next(iter(previous_revisions.values()))
            if ie._unchanged(previous_ie):
                ie.revision = previous_ie.revision
                return
        if ie.has_text():
            with self.branch.repository._text_store.get(ie.text_id) as f:
                file_lines = f.readlines()
            w.add_lines(rev_id, previous_revisions, file_lines)
            self.text_count += 1
        else:
            w.add_lines(rev_id, previous_revisions, [])
        ie.revision = rev_id

    def _make_order(self):
        """Return a suitable order for importing revisions.

        The order must be such that an revision is imported after all
        its (present) parents.
        """
        todo = set(self.revisions)
        done = self.absent_revisions.copy()
        order = []
        while todo:
            for rev_id in sorted(list(todo)):
                rev = self.revisions[rev_id]
                parent_ids = set(rev.parent_ids)
                if parent_ids.issubset(done):
                    order.append(rev_id)
                    todo.remove(rev_id)
                    done.add(rev_id)
        return order