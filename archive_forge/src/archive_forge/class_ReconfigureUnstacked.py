from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class ReconfigureUnstacked:

    def apply(self, controldir):
        branch = controldir.open_branch()
        with branch.lock_write():
            branch.set_stacked_on_url(None)
            if not trace.is_quiet():
                ui.ui_factory.note(gettext('%s is now not stacked\n') % (branch.base,))