from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
@classmethod
def to_standalone(klass, controldir):
    """Convert a repository branch into a standalone branch"""
    reconfiguration = klass(controldir)
    reconfiguration._set_use_shared(use_shared=False)
    if not reconfiguration.changes_planned():
        raise AlreadyStandalone(controldir)
    return reconfiguration