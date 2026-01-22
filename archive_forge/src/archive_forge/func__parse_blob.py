from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _parse_blob(self):
    """Parse a blob command."""
    lineno = self.lineno
    mark = self._get_mark_if_any()
    data = self._get_data(b'blob')
    return commands.BlobCommand(mark, data, lineno)