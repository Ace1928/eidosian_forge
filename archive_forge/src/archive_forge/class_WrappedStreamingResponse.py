from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from google.api_core import bidi
from google.rpc import error_details_pb2
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.calliope import base
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_proxy_types
import grpc
from six.moves import urllib
import socks
class WrappedStreamingResponse(grpc.Call, grpc.Future):
    """Wrapped streaming response.

  Attributes:
    _response: A grpc.Call/grpc.Future instance representing a service response.
    _fn: Function called on each iteration of this iterator. Takes a lambda
         that produces the next response in the _response iterator.
  """

    def __init__(self, response, fn):
        self._response = response
        self._fn = fn

    def initial_metadata(self):
        return self._response.initial_metadata()

    def trailing_metadata(self):
        return self._response.trailing_metadata()

    def code(self):
        return self._response.code()

    def details(self):
        return self._response.details()

    def debug_error_string(self):
        return self._response.debug_error_string()

    def cancel(self):
        return self._response.cancel()

    def cancelled(self):
        return self._response.cancelled()

    def running(self):
        return self._response.running()

    def done(self):
        return self._response.done()

    def result(self, timeout=None):
        return self._response.result(timeout=timeout)

    def exception(self, timeout=None):
        return self._response.exception(timeout=timeout)

    def traceback(self, timeout=None):
        return self._response.traceback(timeout=timeout)

    def add_done_callback(self, fn):
        return self._response.add_done_callback(fn)

    def add_callback(self, callback):
        return self._response.add_callback(callback)

    def is_active(self):
        return self._response.is_active()

    def time_remaining(self):
        return self._response.time_remaining()

    def __iter__(self):
        return self

    def __next__(self):
        return self._fn(lambda: next(self._response))