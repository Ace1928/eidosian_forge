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
def _extract_member(self, tarinfo, targetpath, set_attrs=True, numeric_owner=False):
    """Extract the TarInfo object tarinfo to a physical
           file called targetpath.
        """
    targetpath = targetpath.rstrip('/')
    targetpath = targetpath.replace('/', os.sep)
    upperdirs = os.path.dirname(targetpath)
    if upperdirs and (not os.path.exists(upperdirs)):
        os.makedirs(upperdirs)
    if tarinfo.islnk() or tarinfo.issym():
        self._dbg(1, '%s -> %s' % (tarinfo.name, tarinfo.linkname))
    else:
        self._dbg(1, tarinfo.name)
    if tarinfo.isreg():
        self.makefile(tarinfo, targetpath)
    elif tarinfo.isdir():
        self.makedir(tarinfo, targetpath)
    elif tarinfo.isfifo():
        self.makefifo(tarinfo, targetpath)
    elif tarinfo.ischr() or tarinfo.isblk():
        self.makedev(tarinfo, targetpath)
    elif tarinfo.islnk() or tarinfo.issym():
        self.makelink(tarinfo, targetpath)
    elif tarinfo.type not in SUPPORTED_TYPES:
        self.makeunknown(tarinfo, targetpath)
    else:
        self.makefile(tarinfo, targetpath)
    if set_attrs:
        self.chown(tarinfo, targetpath, numeric_owner)
        if not tarinfo.issym():
            self.chmod(tarinfo, targetpath)
            self.utime(tarinfo, targetpath)