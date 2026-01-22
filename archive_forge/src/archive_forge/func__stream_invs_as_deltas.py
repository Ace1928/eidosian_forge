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
def _stream_invs_as_deltas(self, revision_ids, delta_versus_null=False):
    """Return a stream of inventory-deltas for the given rev ids.

        :param revision_ids: The list of inventories to transmit
        :param delta_versus_null: Don't try to find a minimal delta for this
            entry, instead compute the delta versus the NULL_REVISION. This
            effectively streams a complete inventory. Used for stuff like
            filling in missing parents, etc.
        """
    from_repo = self.from_repository
    revision_keys = [(rev_id,) for rev_id in revision_ids]
    parent_map = from_repo.inventories.get_parent_map(revision_keys)
    inventories = self.from_repository.iter_inventories(revision_ids, 'topological')
    format = from_repo._format
    invs_sent_so_far = {_mod_revision.NULL_REVISION}
    inventory_cache = lru_cache.LRUCache(50)
    null_inventory = from_repo.revision_tree(_mod_revision.NULL_REVISION).root_inventory
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=format.rich_root_data, tree_references=format.supports_tree_reference)
    for inv in inventories:
        key = (inv.revision_id,)
        parent_keys = parent_map.get(key, ())
        delta = None
        if not delta_versus_null and parent_keys:
            parent_ids = [parent_key[0] for parent_key in parent_keys]
            for parent_id in parent_ids:
                if parent_id not in invs_sent_so_far:
                    continue
                if parent_id == _mod_revision.NULL_REVISION:
                    parent_inv = null_inventory
                else:
                    parent_inv = inventory_cache.get(parent_id, None)
                    if parent_inv is None:
                        parent_inv = from_repo.get_inventory(parent_id)
                candidate_delta = inv._make_delta(parent_inv)
                if delta is None or len(delta) > len(candidate_delta):
                    delta = candidate_delta
                    basis_id = parent_id
        if delta is None:
            basis_id = _mod_revision.NULL_REVISION
            delta = inv._make_delta(null_inventory)
        invs_sent_so_far.add(inv.revision_id)
        inventory_cache[inv.revision_id] = inv
        delta_serialized = serializer.delta_to_lines(basis_id, key[-1], delta)
        yield versionedfile.ChunkedContentFactory(key, parent_keys, None, delta_serialized, chunks_are_lines=True)