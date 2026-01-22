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
class _ProtocolGraphWalker:
    """A graph walker that knows the git protocol.

    As a graph walker, this class implements ack(), next(), and reset(). It
    also contains some base methods for interacting with the wire and walking
    the commit tree.

    The work of determining which acks to send is passed on to the
    implementation instance stored in _impl. The reason for this is that we do
    not know at object creation time what ack level the protocol requires. A
    call to set_ack_type() is required to set up the implementation, before
    any calls to next() or ack() are made.
    """

    def __init__(self, handler, object_store: ObjectContainer, get_peeled, get_symrefs) -> None:
        self.handler = handler
        self.store: ObjectContainer = object_store
        self.get_peeled = get_peeled
        self.get_symrefs = get_symrefs
        self.proto = handler.proto
        self.stateless_rpc = handler.stateless_rpc
        self.advertise_refs = handler.advertise_refs
        self._wants: List[bytes] = []
        self.shallow: Set[bytes] = set()
        self.client_shallow: Set[bytes] = set()
        self.unshallow: Set[bytes] = set()
        self._cached = False
        self._cache: List[bytes] = []
        self._cache_index = 0
        self._impl = None

    def determine_wants(self, heads, depth=None):
        """Determine the wants for a set of heads.

        The given heads are advertised to the client, who then specifies which
        refs they want using 'want' lines. This portion of the protocol is the
        same regardless of ack type, and in fact is used to set the ack type of
        the ProtocolGraphWalker.

        If the client has the 'shallow' capability, this method also reads and
        responds to the 'shallow' and 'deepen' lines from the client. These are
        not part of the wants per se, but they set up necessary state for
        walking the graph. Additionally, later code depends on this method
        consuming everything up to the first 'have' line.

        Args:
          heads: a dict of refname->SHA1 to advertise
        Returns: a list of SHA1s requested by the client
        """
        symrefs = self.get_symrefs()
        values = set(heads.values())
        if self.advertise_refs or not self.stateless_rpc:
            for i, (ref, sha) in enumerate(sorted(heads.items())):
                try:
                    peeled_sha = self.get_peeled(ref)
                except KeyError:
                    continue
                if i == 0:
                    logger.info('Sending capabilities: %s', self.handler.capabilities())
                    line = format_ref_line(ref, sha, self.handler.capabilities() + symref_capabilities(symrefs.items()))
                else:
                    line = format_ref_line(ref, sha)
                self.proto.write_pkt_line(line)
                if peeled_sha != sha:
                    self.proto.write_pkt_line(format_ref_line(ref + PEELED_TAG_SUFFIX, peeled_sha))
            self.proto.write_pkt_line(None)
            if self.advertise_refs:
                return []
        want = self.proto.read_pkt_line()
        if not want:
            return []
        line, caps = extract_want_line_capabilities(want)
        self.handler.set_client_capabilities(caps)
        self.set_ack_type(ack_type(caps))
        allowed = (COMMAND_WANT, COMMAND_SHALLOW, COMMAND_DEEPEN, None)
        command, sha = _split_proto_line(line, allowed)
        want_revs = []
        while command == COMMAND_WANT:
            if sha not in values:
                raise GitProtocolError('Client wants invalid object %s' % sha)
            want_revs.append(sha)
            command, sha = self.read_proto_line(allowed)
        self.set_wants(want_revs)
        if command in (COMMAND_SHALLOW, COMMAND_DEEPEN):
            self.unread_proto_line(command, sha)
            self._handle_shallow_request(want_revs)
        if self.stateless_rpc and self.proto.eof():
            return []
        return want_revs

    def unread_proto_line(self, command, value):
        if isinstance(value, int):
            value = str(value).encode('ascii')
        self.proto.unread_pkt_line(command + b' ' + value)

    def nak(self):
        pass

    def ack(self, have_ref):
        if len(have_ref) != 40:
            raise ValueError('invalid sha %r' % have_ref)
        return self._impl.ack(have_ref)

    def reset(self):
        self._cached = True
        self._cache_index = 0

    def next(self):
        if not self._cached:
            if not self._impl and self.stateless_rpc:
                return None
            return next(self._impl)
        self._cache_index += 1
        if self._cache_index > len(self._cache):
            return None
        return self._cache[self._cache_index]
    __next__ = next

    def read_proto_line(self, allowed):
        """Read a line from the wire.

        Args:
          allowed: An iterable of command names that should be allowed.
        Returns: A tuple of (command, value); see _split_proto_line.

        Raises:
          UnexpectedCommandError: If an error occurred reading the line.
        """
        return _split_proto_line(self.proto.read_pkt_line(), allowed)

    def _handle_shallow_request(self, wants):
        while True:
            command, val = self.read_proto_line((COMMAND_DEEPEN, COMMAND_SHALLOW))
            if command == COMMAND_DEEPEN:
                depth = val
                break
            self.client_shallow.add(val)
        self.read_proto_line((None,))
        shallow, not_shallow = _find_shallow(self.store, wants, depth)
        self.shallow.update(shallow - not_shallow)
        new_shallow = self.shallow - self.client_shallow
        unshallow = self.unshallow = not_shallow & self.client_shallow
        for sha in sorted(new_shallow):
            self.proto.write_pkt_line(format_shallow_line(sha))
        for sha in sorted(unshallow):
            self.proto.write_pkt_line(format_unshallow_line(sha))
        self.proto.write_pkt_line(None)

    def notify_done(self):
        self.handler.notify_done()

    def send_ack(self, sha, ack_type=b''):
        self.proto.write_pkt_line(format_ack_line(sha, ack_type))

    def send_nak(self):
        self.proto.write_pkt_line(NAK_LINE)

    def handle_done(self, done_required, done_received):
        return self._impl.handle_done(done_required, done_received)

    def set_wants(self, wants):
        self._wants = wants

    def all_wants_satisfied(self, haves):
        """Check whether all the current wants are satisfied by a set of haves.

        Args:
          haves: A set of commits we know the client has.
        Note: Wants are specified with set_wants rather than passed in since
            in the current interface they are determined outside this class.
        """
        return _all_wants_satisfied(self.store, haves, self._wants)

    def set_ack_type(self, ack_type):
        impl_classes = {MULTI_ACK: MultiAckGraphWalkerImpl, MULTI_ACK_DETAILED: MultiAckDetailedGraphWalkerImpl, SINGLE_ACK: SingleAckGraphWalkerImpl}
        self._impl = impl_classes[ack_type](self)