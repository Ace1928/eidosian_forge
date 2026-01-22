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
def _load_old_inventory(self, rev_id):
    with self.branch.repository.inventory_store.get(rev_id) as f:
        inv = xml4.serializer_v4.read_inventory(f)
    inv.revision_id = rev_id
    rev = self.revisions[rev_id]
    return inv