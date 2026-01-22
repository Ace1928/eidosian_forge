import logging
import numbers
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from typing import AnyStr, Sequence  # noqa
from kombu.log import LOG_LEVELS
from kombu.log import get_logger as _get_logger
from kombu.utils.encoding import safe_str
from .term import colored
class LoggingProxy:
    """Forward file object to :class:`logging.Logger` instance.

    Arguments:
        logger (~logging.Logger): Logger instance to forward to.
        loglevel (int, str): Log level to use when logging messages.
    """
    mode = 'w'
    name = None
    closed = False
    loglevel = logging.ERROR
    _thread = threading.local()

    def __init__(self, logger, loglevel=None):
        self.logger = logger
        self.loglevel = mlevel(loglevel or self.logger.level or self.loglevel)
        self._safewrap_handlers()

    def _safewrap_handlers(self):

        def wrap_handler(handler):

            class WithSafeHandleError(logging.Handler):

                def handleError(self, record):
                    try:
                        traceback.print_exc(None, sys.__stderr__)
                    except OSError:
                        pass
            handler.handleError = WithSafeHandleError().handleError
        return [wrap_handler(h) for h in self.logger.handlers]

    def write(self, data):
        """Write message to logging object."""
        if _in_sighandler:
            safe_data = safe_str(data)
            print(safe_data, file=sys.__stderr__)
            return len(safe_data)
        if getattr(self._thread, 'recurse_protection', False):
            return 0
        if data and (not self.closed):
            self._thread.recurse_protection = True
            try:
                safe_data = safe_str(data).rstrip('\n')
                if safe_data:
                    self.logger.log(self.loglevel, safe_data)
                    return len(safe_data)
            finally:
                self._thread.recurse_protection = False
        return 0

    def writelines(self, sequence):
        """Write list of strings to file.

        The sequence can be any iterable object producing strings.
        This is equivalent to calling :meth:`write` for each string.
        """
        for part in sequence:
            self.write(part)

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def isatty(self):
        """Here for file support."""
        return False