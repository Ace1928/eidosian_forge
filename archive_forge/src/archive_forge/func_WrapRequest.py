from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def WrapRequest(self, http_client, handlers, exc_handler=None, exc_type=Exception, response_encoding=None):
    """Wraps an http client with request modifiers.

    Args:
      http_client: The original http client to be wrapped.
      handlers: [Handler], The handlers to execute before and after the original
        request.
      exc_handler: f(e), A function that takes an exception and handles it. It
        should also throw an exception if you don't want it to be swallowed.
      exc_type: The type of exception that should be caught and given to the
        handler. It could be a tuple to catch more than one exception type.
      response_encoding: str, the encoding to use to decode the response.
    """
    orig_request = http_client.request

    def WrappedRequest(*args, **kwargs):
        """Replacement http_client.request() method."""
        handler_request = self.request_class.FromRequestArgs(*args, **kwargs)
        headers = {h: v for h, v in six.iteritems(handler_request.headers)}
        handler_request.headers = {}
        for h, v in six.iteritems(headers):
            h, v = _EncodeHeader(h, v)
            handler_request.headers[h] = v
        modifier_data = []
        for handler in handlers:
            modifier_result = handler.request(handler_request)
            modifier_data.append(modifier_result)
        try:
            modified_args, modified_kwargs = handler_request.ToRequestArgs()
            response = orig_request(*modified_args, **modified_kwargs)
        except exc_type as e:
            response = None
            if exc_handler:
                exc_handler(e)
                return
            else:
                raise
        if response_encoding is not None:
            response = self.DecodeResponse(response, response_encoding)
        handler_response = self.response_class.FromResponse(response)
        for handler, data in zip(handlers, modifier_data):
            if handler.response:
                handler.response(handler_response, data)
        return response
    http_client.request = WrappedRequest