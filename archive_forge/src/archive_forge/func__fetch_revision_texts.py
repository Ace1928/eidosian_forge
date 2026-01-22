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
def _fetch_revision_texts(self, revs):
    from_sf = self.from_repository.signatures
    keys = [(rev_id,) for rev_id in revs]
    signatures = versionedfile.filter_absent(from_sf.get_record_stream(keys, self.to_format._fetch_order, not self.to_format._fetch_uses_deltas))
    from_rf = self.from_repository.revisions
    revisions = from_rf.get_record_stream(keys, self.to_format._fetch_order, not self.delta_on_metadata())
    return [('signatures', signatures), ('revisions', revisions)]