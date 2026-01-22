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
def _require_root_change(self, tree):
    """Enforce an appropriate root object change.

        This is called once when record_iter_changes is called, if and only if
        the root was not in the delta calculated by record_iter_changes.

        :param tree: The tree which is being committed.
        """
    if self.repository.supports_rich_root():
        return
    if len(self.parents) == 0:
        raise errors.RootMissing()
    entry = entry_factory['directory'](tree.path2id(''), '', None)
    entry.revision = self._new_revision_id
    self._basis_delta.append(('', '', entry.file_id, entry))