from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def GetStub(self, service):
    """Retrieve the stub registered for the specified service.

    Args:
      service: string

    Returns:
      stub

    Returns the stub registered for 'service', and returns the default stub
    if no such stub is found.
    """
    return self.__stub_map.get(service, self.__default_stub)