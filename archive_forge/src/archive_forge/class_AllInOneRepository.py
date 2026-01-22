import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
class AllInOneRepository(VersionedFileRepository):
    """Legacy support - the repository behaviour for all-in-one branches."""

    @property
    def _serializer(self):
        return xml5.serializer_v5

    def _escape(self, file_or_path):
        if not isinstance(file_or_path, str):
            file_or_path = '/'.join(file_or_path)
        if file_or_path == '':
            return ''
        return urlutils.escape(osutils.safe_unicode(file_or_path))

    def __init__(self, _format, a_controldir):
        dir_mode = a_controldir._get_dir_mode()
        file_mode = a_controldir._get_file_mode()

        def get_store(name, compressed=True, prefixed=False):
            relpath = self._escape(name)
            store = TextStore(a_controldir.transport.clone(relpath), prefixed=prefixed, compressed=compressed, dir_mode=dir_mode, file_mode=file_mode)
            return store
        if isinstance(_format, RepositoryFormat4):
            self.inventory_store = get_store('inventory-store')
            self._text_store = get_store('text-store')
        super().__init__(_format, a_controldir, a_controldir._control_files)

    def _all_possible_ids(self):
        """Return all the possible revisions that we could find."""
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(3, '_all_possible_ids scales with size of history.')
        with self.lock_read():
            return [key[-1] for key in self.inventories.keys()]

    def _all_revision_ids(self):
        """Returns a list of all the revision ids in the repository.

        These are in as much topological order as the underlying store can
        present: for weaves ghosts may lead to a lack of correctness until
        the reweave updates the parents list.
        """
        with self.lock_read():
            return [key[-1] for key in self.revisions.keys()]

    def _activate_new_inventory(self):
        """Put a replacement inventory.new into use as inventories."""
        t = self.controldir._control_files._transport
        t.copy('inventory.new.weave', 'inventory.weave')
        t.delete('inventory.new.weave')
        self.inventories.keys()

    def _backup_inventory(self):
        t = self.controldir._control_files._transport
        t.copy('inventory.weave', 'inventory.backup.weave')

    def _temp_inventories(self):
        t = self.controldir._control_files._transport
        return self._format._get_inventories(t, self, 'inventory.new')

    def get_commit_builder(self, branch, parents, config, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False):
        self._check_ascii_revisionid(revision_id, self.get_commit_builder)
        result = VersionedFileCommitBuilder(self, parents, config, timestamp, timezone, committer, revprops, revision_id, lossy=lossy)
        self.start_write_group()
        return result

    def _inventory_add_lines(self, revision_id, parents, lines, check_content=True):
        """Store lines in inv_vf and return the sha1 of the inventory."""
        present_parents = self.get_graph().get_parent_map(parents)
        final_parents = []
        for parent in parents:
            if parent in present_parents:
                final_parents.append((parent,))
        return self.inventories.add_lines((revision_id,), final_parents, lines, check_content=check_content)[0]

    def is_shared(self):
        """AllInOne repositories cannot be shared."""
        return False

    def set_make_working_trees(self, new_value):
        """Set the policy flag for making working trees when creating branches.

        This only applies to branches that use this repository.

        The default is 'True'.
        :param new_value: True to restore the default, False to disable making
                          working trees.
        """
        raise errors.RepositoryUpgradeRequired(self.user_url)

    def make_working_trees(self):
        """Returns the policy for making working trees on new branches."""
        return True