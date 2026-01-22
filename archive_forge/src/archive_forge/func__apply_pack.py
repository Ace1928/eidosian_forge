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
def _apply_pack(self, refs: List[Tuple[bytes, bytes, bytes]]) -> List[Tuple[bytes, bytes]]:
    all_exceptions = (IOError, OSError, ChecksumMismatch, ApplyDeltaError, AssertionError, socket.error, zlib.error, ObjectFormatException)
    status = []
    will_send_pack = False
    for command in refs:
        if command[1] != ZERO_SHA:
            will_send_pack = True
    if will_send_pack:
        try:
            recv = getattr(self.proto, 'recv', None)
            self.repo.object_store.add_thin_pack(self.proto.read, recv)
            status.append((b'unpack', b'ok'))
        except all_exceptions as e:
            status.append((b'unpack', str(e).replace('\n', '').encode('utf-8')))
    else:
        status.append((b'unpack', b'ok'))
    for oldsha, sha, ref in refs:
        ref_status = b'ok'
        try:
            if sha == ZERO_SHA:
                if CAPABILITY_DELETE_REFS not in self.capabilities():
                    raise GitProtocolError('Attempted to delete refs without delete-refs capability.')
                try:
                    self.repo.refs.remove_if_equals(ref, oldsha)
                except all_exceptions:
                    ref_status = b'failed to delete'
            else:
                try:
                    self.repo.refs.set_if_equals(ref, oldsha, sha)
                except all_exceptions:
                    ref_status = b'failed to write'
        except KeyError:
            ref_status = b'bad ref'
        status.append((ref, ref_status))
    return status