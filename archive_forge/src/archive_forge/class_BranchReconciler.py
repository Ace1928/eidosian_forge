from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
class BranchReconciler:
    """Reconciler that works on a branch."""

    def __init__(self, a_branch, thorough=False):
        self.fixed_history = None
        self.thorough = thorough
        self.branch = a_branch

    def reconcile(self):
        with self.branch.lock_write(), ui.ui_factory.nested_progress_bar() as self.pb:
            ret = ReconcileResult()
            ret.fixed_history = self._reconcile_steps()
            return ret

    def _reconcile_steps(self):
        return self._reconcile_revision_history()

    def _reconcile_revision_history(self):
        last_revno, last_revision_id = self.branch.last_revision_info()
        real_history = []
        graph = self.branch.repository.get_graph()
        try:
            for revid in graph.iter_lefthand_ancestry(last_revision_id, (_mod_revision.NULL_REVISION,)):
                real_history.append(revid)
        except errors.RevisionNotPresent:
            pass
        real_history.reverse()
        if last_revno != len(real_history):
            ui.ui_factory.note(gettext('Fixing last revision info {0}  => {1}').format(last_revno, len(real_history)))
            self.branch.set_last_revision_info(len(real_history), last_revision_id)
            return True
        else:
            ui.ui_factory.note(gettext('revision_history ok.'))
            return False