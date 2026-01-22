import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def _urljoin(base, url):
    """Custom urljoin replacement supporting : before / in url."""
    if url.startswith('http://') or url.startswith('https://'):
        return urllib.parse.urljoin(base, url)
    new_base = base if base.endswith('/') else base + '/'
    new_url = url[1:] if url.startswith('/') else url
    return new_base + new_url