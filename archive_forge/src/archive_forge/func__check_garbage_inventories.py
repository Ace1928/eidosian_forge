from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _check_garbage_inventories(self):
    """Check for garbage inventories which we cannot trust

        We cant trust them because their pre-requisite file data may not
        be present - all we know is that their revision was not installed.
        """
    if not self.thorough:
        return
    inventories = set(self.inventory.keys())
    revisions = set(self.revisions.keys())
    garbage = inventories.difference(revisions)
    self.garbage_inventories = len(garbage)
    for revision_key in garbage:
        mutter('Garbage inventory {%s} found.', revision_key[-1])