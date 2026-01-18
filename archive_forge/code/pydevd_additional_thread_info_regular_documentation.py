from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame import PyDBFrame

        Gets the topmost frame for the given thread. Note that it may be None
        and callers should remove the reference to the frame as soon as possible
        to avoid disturbing user code.
        