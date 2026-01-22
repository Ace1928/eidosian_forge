from . import errors, ui
from .i18n import gettext
from .trace import mutter
class Reconciler:
    """Reconcilers are used to reconcile existing data."""

    def __init__(self, dir, other=None, canonicalize_chks=False):
        """Create a Reconciler."""
        self.controldir = dir
        self.canonicalize_chks = canonicalize_chks

    def reconcile(self):
        """Perform reconciliation.
        """
        with ui.ui_factory.nested_progress_bar() as self.pb:
            result = ReconcileResult()
            branch_result = self._reconcile_branch()
            repo_result = self._reconcile_repository()
            result.inconsistent_parents = getattr(repo_result, 'inconsistent_parents', None)
            result.aborted = getattr(repo_result, 'aborted', None)
            result.garbage_inventories = getattr(repo_result, 'garbage_inventories', None)
            result.fixed_branch_history = getattr(branch_result, 'fixed_history', None)
            return result

    def _reconcile_branch(self):
        try:
            self.branch = self.controldir.open_branch()
        except errors.NotBranchError:
            return
        ui.ui_factory.note(gettext('Reconciling branch %s') % self.branch.base)
        return self.branch.reconcile(thorough=True)

    def _reconcile_repository(self):
        self.repo = self.controldir.find_repository()
        ui.ui_factory.note(gettext('Reconciling repository %s') % self.repo.user_url)
        self.pb.update(gettext('Reconciling repository'), 0, 1)
        if self.canonicalize_chks:
            try:
                self.repo.reconcile_canonicalize_chks
            except AttributeError:
                raise errors.BzrError(gettext('%s cannot canonicalize CHKs.') % (self.repo,))
            reconcile_result = self.repo.reconcile_canonicalize_chks()
        else:
            reconcile_result = self.repo.reconcile(thorough=True)
        if reconcile_result.aborted:
            ui.ui_factory.note(gettext('Reconcile aborted: revision index has inconsistent parents.'))
            ui.ui_factory.note(gettext('Run "brz check" for more details.'))
        else:
            ui.ui_factory.note(gettext('Reconciliation complete.'))
        return reconcile_result