import contextlib
import threading
import warnings
class DeviceSynchronized(RuntimeError):
    """Raised when device synchronization is detected while disallowed.

    .. warning::

       This API has been deprecated in CuPy v10 and will be removed in future
       releases.

    .. seealso:: :func:`cupyx.allow_synchronize`

    """

    def __init__(self, message=None):
        if message is None:
            message = 'Device synchronization was detected while disallowed.'
        super().__init__(message)