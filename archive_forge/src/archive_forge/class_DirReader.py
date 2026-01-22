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
class DirReader:
    """An interface for reading directories."""

    def top_prefix_to_starting_dir(self, top, prefix=''):
        """Converts top and prefix to a starting dir entry

        :param top: A utf8 path
        :param prefix: An optional utf8 path to prefix output relative paths
            with.
        :return: A tuple starting with prefix, and ending with the native
            encoding of top.
        """
        raise NotImplementedError(self.top_prefix_to_starting_dir)

    def read_dir(self, prefix, top):
        """Read a specific dir.

        :param prefix: A utf8 prefix to be preprended to the path basenames.
        :param top: A natively encoded path to read.
        :return: A list of the directories contents. Each item contains:
            (utf8_relpath, utf8_name, kind, lstatvalue, native_abspath)
        """
        raise NotImplementedError(self.read_dir)