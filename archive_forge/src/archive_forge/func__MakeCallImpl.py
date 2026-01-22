from __future__ import absolute_import
import sys
from googlecloudsdk.third_party.appengine._internal import six_subset
def _MakeCallImpl(self):
    """Override this method to implement a real asynchronous call rpc."""
    self._state = RPC.RUNNING