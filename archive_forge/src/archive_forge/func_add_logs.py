import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def add_logs(self, logs):
    """Record a log and print it.

    The log should be a tuple `(severity, lineno, col_offset, msg)`, which will
    be printed and recorded. It is part of the log available in the `self.log`
    property.

    Args:
      logs: The logs to add. Must be a list of tuples
        `(severity, lineno, col_offset, msg)`.
    """
    self._log.extend(logs)
    for log in logs:
        print('%s line %d:%d: %s' % log)