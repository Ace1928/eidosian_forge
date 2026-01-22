from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import inspect
import io
from google.auth.transport import requests as google_auth_requests
from google.auth.transport.requests import _MutualTlsOffloadAdapter
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import httplib2
import requests
import six
from six.moves import http_client as httplib
from six.moves import urllib
import socks
from urllib3.util.ssl_ import create_urllib3_context
def GetApitoolsRequests(session, response_handler=None, response_encoding=None):
    """Returns an authenticated httplib2.Http-like object for use by apitools."""
    http_client = _ApitoolsRequests(session, response_handler, response_encoding)
    if hasattr(session, '_googlecloudsdk_credentials'):
        creds = _GoogleAuthApitoolsCredentials(session._googlecloudsdk_credentials)
        orig_request_method = http_client.request

        def HttpRequest(*args, **kwargs):
            return orig_request_method(*args, **kwargs)
        http_client.request = HttpRequest
        setattr(http_client.request, 'credentials', creds)
    return http_client