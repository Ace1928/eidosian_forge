from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _proc_gnusparse_00(self, next, pax_headers, buf):
    """Process a GNU tar extended sparse header, version 0.0.
        """
    offsets = []
    for match in re.finditer(b'\\d+ GNU.sparse.offset=(\\d+)\\n', buf):
        offsets.append(int(match.group(1)))
    numbytes = []
    for match in re.finditer(b'\\d+ GNU.sparse.numbytes=(\\d+)\\n', buf):
        numbytes.append(int(match.group(1)))
    next.sparse = list(zip(offsets, numbytes))