import contextlib
import threading
import warnings
@contextlib.contextmanager
def allow_synchronize(allow):
    """Allows or disallows device synchronization temporarily in the current thread.

    .. warning::

       This API has been deprecated in CuPy v10 and will be removed in future
       releases.

    If device synchronization is detected, :class:`cupyx.DeviceSynchronized`
    will be raised.

    Note that there can be false negatives and positives.
    Device synchronization outside CuPy will not be detected.
    """
    warnings.warn('cupyx.allow_synchronize will be removed in future releases as it is not possible to reliably detect synchronizations.')
    old = _is_allowed()
    _thread_local.allowed = allow
    try:
        yield
    finally:
        _thread_local.allowed = old