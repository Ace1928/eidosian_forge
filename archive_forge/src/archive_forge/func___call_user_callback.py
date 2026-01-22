from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def __call_user_callback(self):
    """Call the high-level callback, if requested."""
    if self.__must_call_user_callback:
        self.__must_call_user_callback = False
        if self.callback is not None:
            self.callback()