import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
def _build_query_string(self, params):
    keys = list(params.keys())
    keys.sort(key=lambda x: x.lower())
    pairs = []
    for key in keys:
        val = get_utf8able_str(params[key])
        pairs.append(key + '=' + self._escape_value(get_utf8able_str(val)))
    return '&'.join(pairs)