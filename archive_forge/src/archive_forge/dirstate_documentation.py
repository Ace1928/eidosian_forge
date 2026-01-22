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
Check that the current entry has a valid parent.

            This makes sure that the parent has a record,
            and that the parent isn't marked as "absent" in the
            current tree. (It is invalid to have a non-absent file in an absent
            directory.)
            