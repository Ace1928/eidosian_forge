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
class Urllib3HttpGitClient(AbstractHttpGitClient):

    def __init__(self, base_url, dumb=None, pool_manager=None, config=None, username=None, password=None, **kwargs) -> None:
        self._username = username
        self._password = password
        if pool_manager is None:
            self.pool_manager = default_urllib3_manager(config, base_url=base_url)
        else:
            self.pool_manager = pool_manager
        if username is not None:
            credentials = f'{username}:{password or ''}'
            import urllib3.util
            basic_auth = urllib3.util.make_headers(basic_auth=credentials)
            self.pool_manager.headers.update(basic_auth)
        self.config = config
        super().__init__(base_url=base_url, dumb=dumb, **kwargs)

    def _get_url(self, path):
        if not isinstance(path, str):
            path = path.decode('utf-8')
        return urljoin(self._base_url, path).rstrip('/') + '/'

    def _http_request(self, url, headers=None, data=None):
        import urllib3.exceptions
        req_headers = self.pool_manager.headers.copy()
        if headers is not None:
            req_headers.update(headers)
        req_headers['Pragma'] = 'no-cache'
        try:
            if data is None:
                resp = self.pool_manager.request('GET', url, headers=req_headers, preload_content=False)
            else:
                resp = self.pool_manager.request('POST', url, headers=req_headers, body=data, preload_content=False)
        except urllib3.exceptions.HTTPError as e:
            raise GitProtocolError(str(e)) from e
        if resp.status == 404:
            raise NotGitRepository
        if resp.status == 401:
            raise HTTPUnauthorized(resp.headers.get('WWW-Authenticate'), url)
        if resp.status == 407:
            raise HTTPProxyUnauthorized(resp.headers.get('Proxy-Authenticate'), url)
        if resp.status != 200:
            raise GitProtocolError('unexpected http resp %d for %s' % (resp.status, url))
        resp.content_type = resp.headers.get('Content-Type')
        try:
            resp_url = resp.geturl()
        except AttributeError:
            resp.redirect_location = resp.get_redirect_location()
        else:
            resp.redirect_location = resp_url if resp_url != url else ''
        return (resp, resp.read)