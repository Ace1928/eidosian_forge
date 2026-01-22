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
def add_trusted_host(self, host: str, source: Optional[str]=None, suppress_logging: bool=False) -> None:
    """
        :param host: It is okay to provide a host that has previously been
            added.
        :param source: An optional source string, for logging where the host
            string came from.
        """
    if not suppress_logging:
        msg = f'adding trusted host: {host!r}'
        if source is not None:
            msg += f' (from {source})'
        logger.info(msg)
    parsed_host, parsed_port = parse_netloc(host)
    if parsed_host is None:
        raise ValueError(f'Trusted host URL must include a host part: {host!r}')
    if (parsed_host, parsed_port) not in self.pip_trusted_origins:
        self.pip_trusted_origins.append((parsed_host, parsed_port))
    self.mount(build_url_from_netloc(host, scheme='http') + '/', self._trusted_host_adapter)
    self.mount(build_url_from_netloc(host) + '/', self._trusted_host_adapter)
    if not parsed_port:
        self.mount(build_url_from_netloc(host, scheme='http') + ':', self._trusted_host_adapter)
        self.mount(build_url_from_netloc(host) + ':', self._trusted_host_adapter)