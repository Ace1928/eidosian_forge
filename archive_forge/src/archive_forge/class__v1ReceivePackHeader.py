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
class _v1ReceivePackHeader:

    def __init__(self, capabilities, old_refs, new_refs) -> None:
        self.want: List[bytes] = []
        self.have: List[bytes] = []
        self._it = self._handle_receive_pack_head(capabilities, old_refs, new_refs)
        self.sent_capabilities = False

    def __iter__(self):
        return self._it

    def _handle_receive_pack_head(self, capabilities, old_refs, new_refs):
        """Handle the head of a 'git-receive-pack' request.

        Args:
          capabilities: List of negotiated capabilities
          old_refs: Old refs, as received from the server
          new_refs: Refs to change

        Returns:
          (have, want) tuple
        """
        self.have = [x for x in old_refs.values() if not x == ZERO_SHA]
        for refname in new_refs:
            if not isinstance(refname, bytes):
                raise TypeError('refname is not a bytestring: %r' % refname)
            old_sha1 = old_refs.get(refname, ZERO_SHA)
            if not isinstance(old_sha1, bytes):
                raise TypeError(f'old sha1 for {refname!r} is not a bytestring: {old_sha1!r}')
            new_sha1 = new_refs.get(refname, ZERO_SHA)
            if not isinstance(new_sha1, bytes):
                raise TypeError(f'old sha1 for {refname!r} is not a bytestring {new_sha1!r}')
            if old_sha1 != new_sha1:
                logger.debug('Sending updated ref %r: %r -> %r', refname, old_sha1, new_sha1)
                if self.sent_capabilities:
                    yield (old_sha1 + b' ' + new_sha1 + b' ' + refname)
                else:
                    yield (old_sha1 + b' ' + new_sha1 + b' ' + refname + b'\x00' + b' '.join(sorted(capabilities)))
                    self.sent_capabilities = True
            if new_sha1 not in self.have and new_sha1 != ZERO_SHA:
                self.want.append(new_sha1)
        yield None