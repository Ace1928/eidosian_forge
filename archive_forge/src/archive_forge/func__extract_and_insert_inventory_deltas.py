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
def _extract_and_insert_inventory_deltas(self, substream, serializer):
    target_rich_root = self.target_repo._format.rich_root_data
    target_tree_refs = self.target_repo._format.supports_tree_reference
    for record in substream:
        inventory_delta_bytes = record.get_bytes_as('lines')
        deserialiser = inventory_delta.InventoryDeltaDeserializer()
        try:
            parse_result = deserialiser.parse_text_bytes(inventory_delta_bytes)
        except inventory_delta.IncompatibleInventoryDelta as err:
            mutter('Incompatible delta: %s', err.msg)
            raise errors.IncompatibleRevision(self.target_repo._format)
        basis_id, new_id, rich_root, tree_refs, inv_delta = parse_result
        revision_id = new_id
        parents = [key[0] for key in record.parents]
        self.target_repo.add_inventory_by_delta(basis_id, inv_delta, revision_id, parents)