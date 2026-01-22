from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _use_pack(self, new_pack):
    """Override _use_pack to check for reconcile having changed content."""
    original_inventory_keys = set()
    inv_index = self._pack_collection.inventory_index.combined_index
    for entry in inv_index.iter_all_entries():
        original_inventory_keys.add(entry[1])
    new_inventory_keys = set()
    for entry in new_pack.inventory_index.iter_all_entries():
        new_inventory_keys.add(entry[1])
    if new_inventory_keys != original_inventory_keys:
        self._data_changed = True
    return new_pack.data_inserted() and self._data_changed