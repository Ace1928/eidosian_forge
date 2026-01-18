from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
@staticmethod
def to_checkout(controldir, bound_location=None):
    """Return a Reconfiguration to convert this controldir into a checkout

        :param controldir: The controldir to reconfigure
        :param bound_location: The location the checkout should be bound to.
        :raise AlreadyCheckout: if controldir is already a checkout
        """
    reconfiguration = Reconfigure(controldir, bound_location)
    reconfiguration._plan_changes(want_tree=True, want_branch=True, want_bound=True, want_reference=False)
    if not reconfiguration.changes_planned():
        raise AlreadyCheckout(controldir)
    return reconfiguration