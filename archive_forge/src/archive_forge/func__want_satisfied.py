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
def _want_satisfied(store: ObjectContainer, haves, want, earliest):
    o = store[want]
    pending = collections.deque([o])
    known = {want}
    while pending:
        commit = pending.popleft()
        if commit.id in haves:
            return True
        if not isinstance(commit, Commit):
            continue
        for parent in commit.parents:
            if parent in known:
                continue
            known.add(parent)
            parent_obj = store[parent]
            assert isinstance(parent_obj, Commit)
            if parent_obj.commit_time >= earliest:
                pending.append(parent_obj)
    return False