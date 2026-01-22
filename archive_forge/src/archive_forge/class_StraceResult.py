import os
import signal
import subprocess
import tempfile
from . import errors
class StraceResult:
    """The result of stracing a function."""

    def __init__(self, raw_log, err_messages):
        """Create a StraceResult.

        :param raw_log: The output that strace created.
        """
        self.raw_log = raw_log
        self.err_messages = err_messages