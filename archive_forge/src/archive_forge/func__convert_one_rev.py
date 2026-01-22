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
def _convert_one_rev(self, rev_id):
    """Convert revision and all referenced objects to new format."""
    rev = self.revisions[rev_id]
    inv = self._load_old_inventory(rev_id)
    present_parents = [p for p in rev.parent_ids if p not in self.absent_revisions]
    self._convert_revision_contents(rev, inv, present_parents)
    self._store_new_inv(rev, inv, present_parents)
    self.converted_revs.add(rev_id)