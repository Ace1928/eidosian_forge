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
@contextlib.contextmanager
def JsonResponseModel(self):
    """In this context, return raw JSON instead of proto."""
    old_model = self.response_type_model
    self.__response_type_model = 'json'
    yield
    self.__response_type_model = old_model