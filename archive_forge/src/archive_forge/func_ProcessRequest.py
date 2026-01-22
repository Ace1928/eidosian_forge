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
def ProcessRequest(self, method_config, request):
    """Hook for pre-processing of requests."""
    if self.log_request:
        logging.info('Calling method %s with %s: %s', method_config.method_id, method_config.request_type_name, request)
    return request