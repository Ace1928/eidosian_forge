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
def _discover_references(self, service, base_url):
    assert base_url[-1] == '/'
    tail = 'info/refs'
    headers = {'Accept': '*/*'}
    if self.dumb is not True:
        tail += '?service=%s' % service.decode('ascii')
    url = urljoin(base_url, tail)
    resp, read = self._http_request(url, headers)
    if resp.redirect_location:
        if not resp.redirect_location.endswith(tail):
            raise GitProtocolError(f'Redirected from URL {url} to URL {resp.redirect_location} without {tail}')
        base_url = urljoin(url, resp.redirect_location[:-len(tail)])
    try:
        self.dumb = resp.content_type is None or not resp.content_type.startswith('application/x-git-')
        if not self.dumb:
            proto = Protocol(read, None)
            try:
                [pkt] = list(proto.read_pkt_seq())
            except ValueError as exc:
                raise GitProtocolError('unexpected number of packets received') from exc
            if pkt.rstrip(b'\n') != b'# service=' + service:
                raise GitProtocolError('unexpected first line %r from smart server' % pkt)
            return (*read_pkt_refs(proto.read_pkt_seq()), base_url)
        else:
            return (read_info_refs(resp), set(), base_url)
    finally:
        resp.close()