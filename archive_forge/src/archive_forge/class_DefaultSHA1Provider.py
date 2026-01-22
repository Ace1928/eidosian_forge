import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
class DefaultSHA1Provider(SHA1Provider):
    """A SHA1Provider that reads directly from the filesystem."""

    def sha1(self, abspath):
        """Return the sha1 of a file given its absolute path."""
        return osutils.sha_file_by_name(abspath)

    def stat_and_sha1(self, abspath):
        """Return the stat and sha1 of a file given its absolute path."""
        with open(abspath, 'rb') as file_obj:
            statvalue = os.fstat(file_obj.fileno())
            sha1 = osutils.sha_file(file_obj)
        return (statvalue, sha1)