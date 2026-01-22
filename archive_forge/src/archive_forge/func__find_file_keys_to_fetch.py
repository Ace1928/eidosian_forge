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
def _find_file_keys_to_fetch(self, revision_ids, pb):
    inv_w = self.inventories
    file_ids = self.fileids_altered_by_revision_ids(revision_ids, inv_w)
    count = 0
    num_file_ids = len(file_ids)
    for file_id, altered_versions in file_ids.items():
        if pb is not None:
            pb.update(gettext('Fetch texts'), count, num_file_ids)
        count += 1
        yield ('file', file_id, altered_versions)