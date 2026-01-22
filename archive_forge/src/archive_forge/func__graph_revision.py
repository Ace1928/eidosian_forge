from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _graph_revision(self, rev_id):
    """Load a revision into the revision graph."""
    self._reweave_step('loading revisions')
    rev = self.repo.get_revision_reconcile(rev_id)
    parents = []
    for parent in rev.parent_ids:
        if self._parent_is_available(parent):
            parents.append(parent)
        else:
            mutter('found ghost %s', parent)
    self._rev_graph[rev_id] = parents