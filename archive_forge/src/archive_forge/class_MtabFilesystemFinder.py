import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
class MtabFilesystemFinder(FilesystemFinder):
    """Find the filesystem for a particular path."""
    MTAB_PATH = '/etc/mtab'

    def __init__(self, mountpoints):

        def key(x):
            return len(x[0])
        self._mountpoints = sorted(mountpoints, key=key, reverse=True)

    @classmethod
    def from_mtab(cls):
        """Create a FilesystemFinder from an mtab-style file.

        Note that this will silenty ignore mtab if it doesn't exist or can not
        be opened.
        """
        try:
            return cls(read_mtab(cls.MTAB_PATH))
        except OSError as e:
            trace.mutter('Unable to read mtab: %s', e)
            return cls([])

    def find(self, path):
        """Find the filesystem used by a particular path.

        :param path: Path to find (bytestring or text type)
        :return: Filesystem name (as text type) or None, if the filesystem is
            unknown.
        """
        if not isinstance(path, bytes):
            path = os.fsencode(path)
        for mountpoint, filesystem in self._mountpoints:
            if is_inside(mountpoint, path):
                return filesystem
        return None