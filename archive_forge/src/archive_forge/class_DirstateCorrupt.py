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
class DirstateCorrupt(errors.BzrError):
    _fmt = 'The dirstate file (%(state)s) appears to be corrupt: %(msg)s'

    def __init__(self, state, msg):
        errors.BzrError.__init__(self)
        self.state = state
        self.msg = msg