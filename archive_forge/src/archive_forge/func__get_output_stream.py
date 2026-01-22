import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def _get_output_stream(destination):
    if destination is None or destination == '-':
        return helpers.binary_stream(getattr(sys.stdout, 'buffer', sys.stdout))
    elif destination.endswith('.gz'):
        import gzip
        return gzip.open(destination, 'wb')
    else:
        return open(destination, 'wb')