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
def _extract_and_insert_revisions(self, substream, serializer):
    for record in substream:
        bytes = record.get_bytes_as('fulltext')
        revision_id = record.key[0]
        rev = serializer.read_revision_from_string(bytes)
        if rev.revision_id != revision_id:
            raise AssertionError('wtf: {} != {}'.format(rev, revision_id))
        self.target_repo.add_revision(revision_id, rev)