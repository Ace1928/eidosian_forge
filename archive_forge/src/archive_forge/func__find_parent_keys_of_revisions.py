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
def _find_parent_keys_of_revisions(self, revision_keys):
    """Similar to _find_parent_ids_of_revisions, but used with keys.

        :param revision_keys: An iterable of revision_keys.
        :return: The parents of all revision_keys that are not already in
            revision_keys
        """
    parent_map = self.revisions.get_parent_map(revision_keys)
    parent_keys = set(itertools.chain.from_iterable(parent_map.values()))
    parent_keys.difference_update(revision_keys)
    parent_keys.discard(_mod_revision.NULL_REVISION)
    return parent_keys