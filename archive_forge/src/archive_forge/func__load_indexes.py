from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _load_indexes(self):
    """Load indexes for the reconciliation."""
    self.transaction = self.repo.get_transaction()
    self.pb.update(gettext('Reading indexes'), 0, 2)
    self.inventory = self.repo.inventories
    self.pb.update(gettext('Reading indexes'), 1, 2)
    self.repo._check_for_inconsistent_revision_parents()
    self.revisions = self.repo.revisions
    self.pb.update(gettext('Reading indexes'), 2, 2)