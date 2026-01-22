from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class MultiRpc(object):
    """A wrapper around multiple UserRPC objects.

  This provides an API similar to that of UserRPC, but wraps multiple
  RPCs such that e.g. .wait() blocks until all wrapped RPCs are
  complete, and .get_result() returns the combined results from all
  wrapped RPCs.

  Class methods:
    flatten(rpcs): Expand a list of UserRPCs and MultiRpcs
      into a list of UserRPCs.
    wait_any(rpcs): Call UserRPC.wait_any(flatten(rpcs)).
    wait_all(rpcs): Call UserRPC.wait_all(flatten(rpcs)).

  Instance methods:
    wait(): Wait for all RPCs.
    check_success(): Wait and then check success for all RPCs.
    get_result(): Wait for all, check successes, then merge
      all results.

  Instance attributes:
    rpcs: The list of wrapped RPCs (returns a copy).
    state: The combined state of all RPCs.
  """

    def __init__(self, rpcs, extra_hook=None):
        """Constructor.

    Args:
      rpcs: A list of UserRPC and MultiRpc objects; it is flattened
        before being stored.
      extra_hook: Optional function to be applied to the final result
        or list of results.
    """
        self.__rpcs = self.flatten(rpcs)
        self.__extra_hook = extra_hook

    @property
    def rpcs(self):
        """Get a flattened list containing the RPCs wrapped.

    This returns a copy to prevent users from modifying the state.
    """
        return list(self.__rpcs)

    @property
    def state(self):
        """Get the combined state of the wrapped RPCs.

    This mimics the UserRPC.state property.  If all wrapped RPCs have
    the same state, that state is returned; otherwise, RUNNING is
    returned (which here really means 'neither fish nor flesh').
    """
        lo = apiproxy_rpc.RPC.FINISHING
        hi = apiproxy_rpc.RPC.IDLE
        for rpc in self.__rpcs:
            lo = min(lo, rpc.state)
            hi = max(hi, rpc.state)
        if lo == hi:
            return lo
        return apiproxy_rpc.RPC.RUNNING

    def wait(self):
        """Wait for all wrapped RPCs to finish.

    This mimics the UserRPC.wait() method.
    """
        apiproxy_stub_map.UserRPC.wait_all(self.__rpcs)

    def check_success(self):
        """Check success of all wrapped RPCs, failing if any of the failed.

    This mimics the UserRPC.check_success() method.

    NOTE: This first waits for all wrapped RPCs to finish before
    checking the success of any of them.  This makes debugging easier.
    """
        self.wait()
        for rpc in self.__rpcs:
            rpc.check_success()

    def get_result(self):
        """Return the combined results of all wrapped RPCs.

    This mimics the UserRPC.get_results() method.  Multiple results
    are combined using the following rules:

    1. If there are no wrapped RPCs, an empty list is returned.

    2. If exactly one RPC is wrapped, its result is returned.

    3. If more than one RPC is wrapped, the result is always a list,
       which is constructed from the wrapped results as follows:

       a. A wrapped result equal to None is ignored;

       b. A wrapped result that is a list (but not any other type of
          sequence!) has its elements added to the result list.

       c. Any other wrapped result is appended to the result list.

    After all results are combined, if __extra_hook is set, it is
    called with the combined results and its return value becomes the
    final result.

    NOTE: This first waits for all wrapped RPCs to finish, and then
    checks all their success.  This makes debugging easier.
    """
        if len(self.__rpcs) == 1:
            results = self.__rpcs[0].get_result()
        else:
            results = []
            for rpc in self.__rpcs:
                result = rpc.get_result()
                if isinstance(result, list):
                    results.extend(result)
                elif result is not None:
                    results.append(result)
        if self.__extra_hook is not None:
            results = self.__extra_hook(results)
        return results

    @classmethod
    def flatten(cls, rpcs):
        """Return a list of UserRPCs, expanding MultiRpcs in the argument list.

    For example: given 4 UserRPCs rpc1 through rpc4,
    flatten(rpc1, MultiRpc([rpc2, rpc3], rpc4)
    returns [rpc1, rpc2, rpc3, rpc4].

    Args:
      rpcs: A list of UserRPC and MultiRpc objects.

    Returns:
      A list of UserRPC objects.
    """
        flat = []
        for rpc in rpcs:
            if isinstance(rpc, MultiRpc):
                flat.extend(rpc.__rpcs)
            else:
                if not isinstance(rpc, apiproxy_stub_map.UserRPC):
                    raise datastore_errors.BadArgumentError('Expected a list of UserRPC object (%r)' % (rpc,))
                flat.append(rpc)
        return flat

    @classmethod
    def wait_any(cls, rpcs):
        """Wait until one of the RPCs passed in is finished.

    This mimics UserRPC.wait_any().

    Args:
      rpcs: A list of UserRPC and MultiRpc objects.

    Returns:
      A UserRPC object or None.
    """
        return apiproxy_stub_map.UserRPC.wait_any(cls.flatten(rpcs))

    @classmethod
    def wait_all(cls, rpcs):
        """Wait until all RPCs passed in are finished.

    This mimics UserRPC.wait_all().

    Args:
      rpcs: A list of UserRPC and MultiRpc objects.
    """
        apiproxy_stub_map.UserRPC.wait_all(cls.flatten(rpcs))