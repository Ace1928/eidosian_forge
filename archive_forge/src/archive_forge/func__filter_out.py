import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def _filter_out(self, raw_dirblocks):
    """Filter out a walkdirs_utf8 result.

        stat field is removed, all native paths are converted to unicode
        """
    filtered_dirblocks = []
    for dirinfo, block in raw_dirblocks:
        dirinfo = (dirinfo[0], self._native_to_unicode(dirinfo[1]))
        details = []
        for line in block:
            details.append(line[0:3] + (self._native_to_unicode(line[4]),))
        filtered_dirblocks.append((dirinfo, details))
    return filtered_dirblocks