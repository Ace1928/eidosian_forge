from io import StringIO
import logging
import unittest
from numba.core import tracing
class CapturedTrace:
    """Capture the trace temporarily for validation."""

    def __init__(self):
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)

    def __enter__(self):
        self._handlers = logger.handlers
        self.buffer = StringIO()
        logger.handlers = [logging.StreamHandler(self.buffer)]

    def __exit__(self, type, value, traceback):
        logger.handlers = self._handlers

    def getvalue(self):
        log = self.buffer.getvalue()
        log = log.replace(__name__ + '.', '')
        return log