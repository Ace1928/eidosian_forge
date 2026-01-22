import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
class ShapelyDeprecationWarning(FutureWarning):
    """
    Warning for features that will be removed or behaviour that will be
    changed in a future release.
    """