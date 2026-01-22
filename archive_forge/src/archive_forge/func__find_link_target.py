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
def _find_link_target(self, tarinfo):
    """Find the target member of a symlink or hardlink member in the
           archive.
        """
    if tarinfo.issym():
        linkname = '/'.join(filter(None, (os.path.dirname(tarinfo.name), tarinfo.linkname)))
        limit = None
    else:
        linkname = tarinfo.linkname
        limit = tarinfo
    member = self._getmember(linkname, tarinfo=limit, normalize=True)
    if member is None:
        raise KeyError('linkname %r not found' % linkname)
    return member