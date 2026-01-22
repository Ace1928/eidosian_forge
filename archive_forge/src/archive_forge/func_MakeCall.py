from __future__ import absolute_import
import sys
from googlecloudsdk.third_party.appengine._internal import six_subset
def MakeCall(self, package=None, call=None, request=None, response=None, callback=None, deadline=None):
    """Makes an asynchronous (i.e. non-blocking) API call within the
    specified package for the specified call method.

    It will call the _MakeRealCall to do the real job.

    Args:
      Same as constructor; see __init__.

    Raises:
      TypeError or AssertionError if an argument is of an invalid type.
      AssertionError or RuntimeError is an RPC is already in use.
    """
    self.callback = callback or self.callback
    self.package = package or self.package
    self.call = call or self.call
    self.request = request or self.request
    self.response = response or self.response
    self.deadline = deadline or self.deadline
    assert self._state is RPC.IDLE, 'RPC for %s.%s has already been started' % (self.package, self.call)
    assert self.callback is None or callable(self.callback)
    self._MakeCallImpl()