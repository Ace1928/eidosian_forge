import os
import re
import sys
import ctypes
import ctypes.util
import pyglet
def _create_ld_so_cache(self):
    directories = []
    try:
        directories.extend(os.environ['LD_LIBRARY_PATH'].split(':'))
    except KeyError:
        pass
    try:
        with open('/etc/ld.so.conf') as fid:
            directories.extend([directory.strip() for directory in fid])
    except IOError:
        pass
    directories.extend(['/lib', '/usr/lib'])
    self._ld_so_cache = self._find_libs(directories)