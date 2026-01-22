import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
def _handle_upload_pack_tail(proto, capabilities: Set[bytes], graph_walker, pack_data: Callable[[bytes], None], progress: Optional[Callable[[bytes], None]]=None, rbufsize=_RBUFSIZE):
    """Handle the tail of a 'git-upload-pack' request.

    Args:
      proto: Protocol object to read from
      capabilities: List of negotiated capabilities
      graph_walker: GraphWalker instance to call .ack() on
      pack_data: Function to call with pack data
      progress: Optional progress reporting function
      rbufsize: Read buffer size
    """
    pkt = proto.read_pkt_line()
    while pkt:
        parts = pkt.rstrip(b'\n').split(b' ')
        if parts[0] == b'ACK':
            graph_walker.ack(parts[1])
        if parts[0] == b'NAK':
            graph_walker.nak()
        if len(parts) < 3 or parts[2] not in (b'ready', b'continue', b'common'):
            break
        pkt = proto.read_pkt_line()
    if CAPABILITY_SIDE_BAND_64K in capabilities:
        if progress is None:

            def progress(x):
                pass
        for chan, data in _read_side_band64k_data(proto.read_pkt_seq()):
            if chan == SIDE_BAND_CHANNEL_DATA:
                pack_data(data)
            elif chan == SIDE_BAND_CHANNEL_PROGRESS:
                progress(data)
            else:
                raise AssertionError('Invalid sideband channel %d' % chan)
    else:
        while True:
            data = proto.read(rbufsize)
            if data == b'':
                break
            pack_data(data)