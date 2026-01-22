from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class ReconfigureStackedOn:
    """Reconfigures a branch to be stacked on another branch."""

    def apply(self, controldir, stacked_on_url):
        branch = controldir.open_branch()
        on_url = urlutils.relative_url(branch.base, urlutils.normalize_url(stacked_on_url))
        with branch.lock_write():
            branch.set_stacked_on_url(on_url)
            if not trace.is_quiet():
                ui.ui_factory.note(gettext('{0} is now stacked on {1}\n').format(branch.base, branch.get_stacked_on_url()))