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
def __FinalUrlValue(self, value, field):
    """Encode value for the URL, using field to skip encoding for bytes."""
    if isinstance(field, messages.BytesField) and value is not None:
        return base64.urlsafe_b64encode(value)
    elif isinstance(value, six.text_type):
        return value.encode('utf8')
    elif isinstance(value, six.binary_type):
        return value.decode('utf8')
    elif isinstance(value, datetime.datetime):
        return value.isoformat()
    return value