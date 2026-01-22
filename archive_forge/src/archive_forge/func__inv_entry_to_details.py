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
@staticmethod
def _inv_entry_to_details(inv_entry):
    """Convert an inventory entry (from a revision tree) to state details.

        :param inv_entry: An inventory entry whose sha1 and link targets can be
            relied upon, and which has a revision set.
        :return: A details tuple - the details for a single tree at a path +
            id.
        """
    kind = inv_entry.kind
    minikind = DirState._kind_to_minikind[kind]
    tree_data = inv_entry.revision
    if kind == 'directory':
        fingerprint = b''
        size = 0
        executable = False
    elif kind == 'symlink':
        if inv_entry.symlink_target is None:
            fingerprint = b''
        else:
            fingerprint = inv_entry.symlink_target.encode('utf8')
        size = 0
        executable = False
    elif kind == 'file':
        fingerprint = inv_entry.text_sha1 or b''
        size = inv_entry.text_size or 0
        executable = inv_entry.executable
    elif kind == 'tree-reference':
        fingerprint = inv_entry.reference_revision or b''
        size = 0
        executable = False
    else:
        raise Exception("can't pack %s" % inv_entry)
    return static_tuple.StaticTuple(minikind, fingerprint, size, executable, tree_data)