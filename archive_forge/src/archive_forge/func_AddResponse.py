from __future__ import absolute_import
import io
import logging
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import AbstractRpcServer
from googlecloudsdk.third_party.appengine.tools.appengine_rpc import HttpRpcServer
from googlecloudsdk.third_party.appengine._internal import six_subset
def AddResponse(self, url, response_func):
    """Calls the provided function when the provided URL is requested.

      The provided function should accept a request object and return a
      response object.

      Args:
        url: The URL to trigger on.
        response_func: The function to call when the url is requested.
      """
    self.responses[url] = response_func