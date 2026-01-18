from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
@classmethod
def wait_any(cls, rpcs):
    """Wait until an RPC is finished.

    Args:
      rpcs: Iterable collection of UserRPC instances.

    Returns:
      A UserRPC instance, indicating the first RPC among the given
      RPCs that finished; or None, indicating that either an RPC not
      among the given RPCs finished in the mean time, or the iterable
      is empty.

    NOTES:

    (1) Repeatedly calling wait_any() with the same arguments will not
        make progress; it will keep returning the same RPC (the one
        that finished first).  The callback, however, will only be
        called the first time the RPC finishes (which may be here or
        in the wait() method).

    (2) It may return before any of the given RPCs finishes, if
        another pending RPC exists that is not included in the rpcs
        argument.  In this case the other RPC's callback will *not*
        be called.  The motivation for this feature is that wait_any()
        may be used as a low-level building block for a variety of
        high-level constructs, some of which prefer to block for the
        minimal amount of time without busy-waiting.
    """
    assert iter(rpcs) is not rpcs, 'rpcs must be a collection, not an iterator'
    finished, running = cls.__check_one(rpcs)
    if finished is not None:
        return finished
    if running is None:
        return None
    try:
        cls.__local.may_interrupt_wait = True
        try:
            running.__rpc.Wait()
        except apiproxy_errors.InterruptedError as err:
            err.rpc._exception = None
            err.rpc._traceback = None
    finally:
        cls.__local.may_interrupt_wait = False
    finished, running = cls.__check_one(rpcs)
    return finished