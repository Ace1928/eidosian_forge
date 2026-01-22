from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _get_proposed_inventory(self, delta):
    if len(self.parents):
        new_inv = self.basis_inventory.create_by_apply_delta(delta, b'not-a-valid-revision-id:')
    else:
        new_inv = inventory.Inventory(revision_id=self.revision_id)
        new_inv.delete(inventory.ROOT_ID)
        try:
            new_inv.apply_delta(delta)
        except errors.InconsistentDelta:
            self.mutter('INCONSISTENT DELTA IS:\n%s' % '\n'.join([str(de) for de in delta]))
            raise
    return new_inv