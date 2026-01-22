import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _control_app_writing(self):
    size = self._get_write_buffer_size()
    if size >= self._outgoing_high_water and (not self._app_writing_paused):
        self._app_writing_paused = True
        try:
            self._app_protocol.pause_writing()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self._loop.call_exception_handler({'message': 'protocol.pause_writing() failed', 'exception': exc, 'transport': self._app_transport, 'protocol': self})
    elif size <= self._outgoing_low_water and self._app_writing_paused:
        self._app_writing_paused = False
        try:
            self._app_protocol.resume_writing()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            self._loop.call_exception_handler({'message': 'protocol.resume_writing() failed', 'exception': exc, 'transport': self._app_transport, 'protocol': self})