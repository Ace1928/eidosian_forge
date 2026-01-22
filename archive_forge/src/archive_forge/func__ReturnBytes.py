import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def _ReturnBytes(unused_http, http_request, *unused_args, **unused_kwds):
    url = http_request.url
    if url == 'https://part.one/':
        return http_wrapper.Response(info={'content-location': 'https://part.two/', 'content-range': 'bytes 0-25/52', 'status': http_client.PARTIAL_CONTENT}, content=string.ascii_lowercase, request_url='https://part.one/')
    elif url == 'https://part.two/':
        return http_wrapper.Response(info={'content-range': 'bytes 26-51/52', 'status': http_client.OK}, content=string.ascii_uppercase, request_url='https://part.two/')
    else:
        self.fail('Unknown URL requested: %s' % url)