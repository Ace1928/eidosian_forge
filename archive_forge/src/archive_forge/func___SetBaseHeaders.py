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
def __SetBaseHeaders(self, http_request, client):
    """Fill in the basic headers on http_request."""
    user_agent = client.user_agent or 'apitools-client/1.0'
    http_request.headers['user-agent'] = user_agent
    http_request.headers['accept'] = 'application/json'
    http_request.headers['accept-encoding'] = 'gzip, deflate'