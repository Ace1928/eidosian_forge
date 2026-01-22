import logging
import logging.handlers
import os
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows
from coloredlogs import (
class SystemLogging(object):
    """Context manager to enable system logging."""

    def __init__(self, *args, **kw):
        """
        Initialize a :class:`SystemLogging` object.

        :param args: Positional arguments to :func:`enable_system_logging()`.
        :param kw: Keyword arguments to :func:`enable_system_logging()`.
        """
        self.args = args
        self.kw = kw
        self.handler = None

    def __enter__(self):
        """Enable system logging when entering the context."""
        if self.handler is None:
            self.handler = enable_system_logging(*self.args, **self.kw)
        return self.handler

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """
        Disable system logging when leaving the context.

        .. note:: If an exception is being handled when we leave the context a
                  warning message including traceback is logged *before* system
                  logging is disabled.
        """
        if self.handler is not None:
            if exc_type is not None:
                logger.warning('Disabling system logging due to unhandled exception!', exc_info=True)
            (self.kw.get('logger') or logging.getLogger()).removeHandler(self.handler)
            self.handler = None