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
class Win32FilesystemFinder(FilesystemFinder):

    def find(self, path):
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        if isinstance(drive, bytes):
            drive = os.fsdecode(drive)
        fs_type = win32utils.get_fs_type(drive + '\\')
        if fs_type is None:
            return None
        return {'FAT32': 'vfat', 'NTFS': 'ntfs'}.get(fs_type, fs_type)