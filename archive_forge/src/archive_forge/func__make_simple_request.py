from distutils.version import StrictVersion
import functools
from http import client as http_client
import json
import logging
import re
import textwrap
import time
from urllib import parse as urlparse
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common.i18n import _
from ironicclient import exc
def _make_simple_request(self, conn, method, url):
    return conn.request(url, method, raise_exc=False, user_agent=USER_AGENT, endpoint_filter=self._get_endpoint_filter(), endpoint_override=self.endpoint_override)