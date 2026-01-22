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
class UploadArchiveHandler(Handler):

    def __init__(self, backend, args, proto, stateless_rpc=False) -> None:
        super().__init__(backend, proto, stateless_rpc)
        self.repo = backend.open_repository(args[0])

    def handle(self):

        def write(x):
            return self.proto.write_sideband(SIDE_BAND_CHANNEL_DATA, x)
        arguments = []
        for pkt in self.proto.read_pkt_seq():
            key, value = pkt.split(b' ', 1)
            if key != b'argument':
                raise GitProtocolError('unknown command %s' % key)
            arguments.append(value.rstrip(b'\n'))
        prefix = b''
        format = 'tar'
        i = 0
        store: ObjectContainer = self.repo.object_store
        while i < len(arguments):
            argument = arguments[i]
            if argument == b'--prefix':
                i += 1
                prefix = arguments[i]
            elif argument == b'--format':
                i += 1
                format = arguments[i].decode('ascii')
            else:
                commit_sha = self.repo.refs[argument]
                tree = store[store[commit_sha].tree]
            i += 1
        self.proto.write_pkt_line(b'ACK')
        self.proto.write_pkt_line(None)
        for chunk in tar_stream(store, tree, mtime=time.time(), prefix=prefix, format=format):
            write(chunk)
        self.proto.write_pkt_line(None)