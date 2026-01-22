import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
def _query_events():
    return self._send_request('get', 'https', path, headers=self._get_auth_header(), params={'limit': 20000, 'timeout': self._query_timeout})