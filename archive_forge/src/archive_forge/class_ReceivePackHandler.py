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
class ReceivePackHandler(PackHandler):
    """Protocol handler for downloading a pack from the client."""

    def __init__(self, backend, args, proto, stateless_rpc=False, advertise_refs=False) -> None:
        super().__init__(backend, proto, stateless_rpc=stateless_rpc)
        self.repo = backend.open_repository(args[0])
        self.advertise_refs = advertise_refs

    @classmethod
    def capabilities(cls) -> Iterable[bytes]:
        return [CAPABILITY_REPORT_STATUS, CAPABILITY_DELETE_REFS, CAPABILITY_QUIET, CAPABILITY_OFS_DELTA, CAPABILITY_SIDE_BAND_64K, CAPABILITY_NO_DONE]

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

    def _report_status(self, status: List[Tuple[bytes, bytes]]) -> None:
        if self.has_capability(CAPABILITY_SIDE_BAND_64K):
            writer = BufferedPktLineWriter(lambda d: self.proto.write_sideband(SIDE_BAND_CHANNEL_DATA, d))
            write = writer.write

            def flush():
                writer.flush()
                self.proto.write_pkt_line(None)
        else:
            write = self.proto.write_pkt_line

            def flush():
                pass
        for name, msg in status:
            if name == b'unpack':
                write(b'unpack ' + msg + b'\n')
            elif msg == b'ok':
                write(b'ok ' + name + b'\n')
            else:
                write(b'ng ' + name + b' ' + msg + b'\n')
        write(None)
        flush()

    def _on_post_receive(self, client_refs):
        hook = self.repo.hooks.get('post-receive', None)
        if not hook:
            return
        try:
            output = hook.execute(client_refs)
            if output:
                self.proto.write_sideband(SIDE_BAND_CHANNEL_PROGRESS, output)
        except HookError as err:
            self.proto.write_sideband(SIDE_BAND_CHANNEL_FATAL, str(err).encode('utf-8'))

    def handle(self) -> None:
        if self.advertise_refs or not self.stateless_rpc:
            refs = sorted(self.repo.get_refs().items())
            symrefs = sorted(self.repo.refs.get_symrefs().items())
            if not refs:
                refs = [(CAPABILITIES_REF, ZERO_SHA)]
            logger.info('Sending capabilities: %s', self.capabilities())
            self.proto.write_pkt_line(format_ref_line(refs[0][0], refs[0][1], self.capabilities() + symref_capabilities(symrefs)))
            for i in range(1, len(refs)):
                ref = refs[i]
                self.proto.write_pkt_line(format_ref_line(ref[0], ref[1]))
            self.proto.write_pkt_line(None)
            if self.advertise_refs:
                return
        client_refs = []
        ref = self.proto.read_pkt_line()
        if ref is None:
            return
        ref, caps = extract_capabilities(ref)
        self.set_client_capabilities(caps)
        while ref:
            client_refs.append(ref.split())
            ref = self.proto.read_pkt_line()
        status = self._apply_pack(client_refs)
        self._on_post_receive(client_refs)
        if self.has_capability(CAPABILITY_REPORT_STATUS):
            self._report_status(status)