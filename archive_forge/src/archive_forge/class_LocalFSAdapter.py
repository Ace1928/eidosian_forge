import email.utils
import io
import ipaddress
import json
import logging
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import warnings
from typing import (
from pip._vendor import requests, urllib3
from pip._vendor.cachecontrol import CacheControlAdapter as _BaseCacheControlAdapter
from pip._vendor.requests.adapters import DEFAULT_POOLBLOCK, BaseAdapter
from pip._vendor.requests.adapters import HTTPAdapter as _BaseHTTPAdapter
from pip._vendor.requests.models import PreparedRequest, Response
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3.connectionpool import ConnectionPool
from pip._vendor.urllib3.exceptions import InsecureRequestWarning
from pip import __version__
from pip._internal.metadata import get_default_environment
from pip._internal.models.link import Link
from pip._internal.network.auth import MultiDomainBasicAuth
from pip._internal.network.cache import SafeFileCache
from pip._internal.utils.compat import has_tls
from pip._internal.utils.glibc import libc_ver
from pip._internal.utils.misc import build_url_from_netloc, parse_netloc
from pip._internal.utils.urls import url_to_path
class LocalFSAdapter(BaseAdapter):

    def send(self, request: PreparedRequest, stream: bool=False, timeout: Optional[Union[float, Tuple[float, float]]]=None, verify: Union[bool, str]=True, cert: Optional[Union[str, Tuple[str, str]]]=None, proxies: Optional[Mapping[str, str]]=None) -> Response:
        pathname = url_to_path(request.url)
        resp = Response()
        resp.status_code = 200
        resp.url = request.url
        try:
            stats = os.stat(pathname)
        except OSError as exc:
            resp.status_code = 404
            resp.reason = type(exc).__name__
            resp.raw = io.BytesIO(f'{resp.reason}: {exc}'.encode('utf8'))
        else:
            modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
            content_type = mimetypes.guess_type(pathname)[0] or 'text/plain'
            resp.headers = CaseInsensitiveDict({'Content-Type': content_type, 'Content-Length': stats.st_size, 'Last-Modified': modified})
            resp.raw = open(pathname, 'rb')
            resp.close = resp.raw.close
        return resp

    def close(self) -> None:
        pass