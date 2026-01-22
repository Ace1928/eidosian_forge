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
def _heads(self, file_id, revision_ids):
    """Calculate the graph heads for revision_ids in the graph of file_id.

        This can use either a per-file graph or a global revision graph as we
        have an identity relationship between the two graphs.
        """
    return self.__heads(revision_ids)