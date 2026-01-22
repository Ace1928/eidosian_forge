import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
def _all_wants_satisfied(store: ObjectContainer, haves, wants):
    """Check whether all the current wants are satisfied by a set of haves.

    Args:
      store: Object store to retrieve objects from
      haves: A set of commits we know the client has.
      wants: A set of commits the client wants
    Note: Wants are specified with set_wants rather than passed in since
        in the current interface they are determined outside this class.
    """
    haves = set(haves)
    if haves:
        have_objs = [store[h] for h in haves]
        earliest = min([h.commit_time for h in have_objs if isinstance(h, Commit)])
    else:
        earliest = 0
    for want in wants:
        if not _want_satisfied(store, haves, want, earliest):
            return False
    return True