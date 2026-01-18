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
def utime(self, tarinfo, targetpath):
    """Set modification time of targetpath according to tarinfo.
        """
    mtime = tarinfo.mtime
    if mtime is None:
        return
    if not hasattr(os, 'utime'):
        return
    try:
        os.utime(targetpath, (mtime, mtime))
    except OSError as e:
        raise ExtractError('could not change modification time') from e