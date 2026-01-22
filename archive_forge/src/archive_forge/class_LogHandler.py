from warnings import warn
import logging
import sys
from rdkit import Chem
from .errors import StopValidateError
from .validations import VALIDATIONS
class LogHandler(logging.Handler):
    """A simple logging Handler that just stores logs in an array until flushed."""

    def __init__(self):
        logging.Handler.__init__(self)
        self.logs = []

    @property
    def logmessages(self):
        return [self.format(record) for record in self.logs]

    def emit(self, record):
        """Append the record."""
        self.logs.append(record)

    def flush(self):
        """Clear the log records."""
        self.acquire()
        try:
            self.logs = []
        finally:
            self.release()

    def close(self):
        """Close the handler."""
        self.flush()
        logging.Handler.close(self)