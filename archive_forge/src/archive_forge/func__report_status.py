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