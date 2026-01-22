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
def _check_inventories(self, checker):
    """Check the inventories found from the revision scan.

        This is responsible for verifying the sha1 of inventories and
        creating a pending_keys set that covers data referenced by inventories.
        """
    with ui.ui_factory.nested_progress_bar() as bar:
        self._do_check_inventories(checker, bar)