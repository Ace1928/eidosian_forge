import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
class _OverlappedFuture(futures.Future):
    """Subclass of Future which represents an overlapped operation.

    Cancelling it will immediately cancel the overlapped operation.
    """

    def __init__(self, ov, *, loop=None):
        super().__init__(loop=loop)
        if self._source_traceback:
            del self._source_traceback[-1]
        self._ov = ov

    def _repr_info(self):
        info = super()._repr_info()
        if self._ov is not None:
            state = 'pending' if self._ov.pending else 'completed'
            info.insert(1, f'overlapped=<{state}, {self._ov.address:#x}>')
        return info

    def _cancel_overlapped(self):
        if self._ov is None:
            return
        try:
            self._ov.cancel()
        except OSError as exc:
            context = {'message': 'Cancelling an overlapped future failed', 'exception': exc, 'future': self}
            if self._source_traceback:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)
        self._ov = None

    def cancel(self, msg=None):
        self._cancel_overlapped()
        return super().cancel(msg=msg)

    def set_exception(self, exception):
        super().set_exception(exception)
        self._cancel_overlapped()

    def set_result(self, result):
        super().set_result(result)
        self._ov = None