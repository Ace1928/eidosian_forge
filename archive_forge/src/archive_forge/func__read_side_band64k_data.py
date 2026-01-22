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
def _read_side_band64k_data(pkt_seq: Iterable[bytes]) -> Iterator[Tuple[int, bytes]]:
    """Read per-channel data.

    This requires the side-band-64k capability.

    Args:
      pkt_seq: Sequence of packets to read
    """
    for pkt in pkt_seq:
        channel = ord(pkt[:1])
        yield (channel, pkt[1:])