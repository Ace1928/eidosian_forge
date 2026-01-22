from __future__ import print_function
import io
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import urllib3
import certifi
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions, json_dumps
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
def _make_pool(self, parsed_dsn, http_proxy, https_proxy, ca_certs, proxy_headers):
    proxy = None
    no_proxy = self._in_no_proxy(parsed_dsn)
    if parsed_dsn.scheme == 'https' and https_proxy != '':
        proxy = https_proxy or (not no_proxy and getproxies().get('https'))
    if not proxy and http_proxy != '':
        proxy = http_proxy or (not no_proxy and getproxies().get('http'))
    opts = self._get_pool_options(ca_certs)
    if proxy:
        if proxy_headers:
            opts['proxy_headers'] = proxy_headers
        if proxy.startswith('socks'):
            use_socks_proxy = True
            try:
                from urllib3.contrib.socks import SOCKSProxyManager
            except ImportError:
                use_socks_proxy = False
                logger.warning('You have configured a SOCKS proxy (%s) but support for SOCKS proxies is not installed. Disabling proxy support. Please add `PySocks` (or `urllib3` with the `[socks]` extra) to your dependencies.', proxy)
            if use_socks_proxy:
                return SOCKSProxyManager(proxy, **opts)
            else:
                return urllib3.PoolManager(**opts)
        else:
            return urllib3.ProxyManager(proxy, **opts)
    else:
        return urllib3.PoolManager(**opts)