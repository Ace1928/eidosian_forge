from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _reweave_step(self, message):
    """Mark a single step of regeneration complete."""
    self.pb.update(message, self.count, self.total)
    self.count += 1