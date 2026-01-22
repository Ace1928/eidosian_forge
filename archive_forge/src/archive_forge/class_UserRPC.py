from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class UserRPC(object):
    """Wrapper class for asynchronous RPC.

  Simplest low-level usage pattern:

    rpc = UserRPC('service', [deadline], [callback])
    rpc.make_call('method', request, response)
    .
    .
    .
    rpc.wait()
    rpc.check_success()

  However, a service module normally provides a wrapper so that the
  typical usage pattern becomes more like this:

    from google.appengine.api import service
    rpc = service.create_rpc([deadline], [callback])
    service.make_method_call(rpc, [service-specific-args])
    .
    .
    .
    rpc.wait()
    result = rpc.get_result()

  The service.make_method_call() function sets a service- and method-
  specific hook function that is called by rpc.get_result() with the
  rpc object as its first argument, and service-specific value as its
  second argument.  The hook function should call rpc.check_success()
  and then extract the user-level result from the rpc.result
  protobuffer.  Additional arguments may be passed from
  make_method_call() to the get_result hook via the second argument.

  Also note wait_any() and wait_all(), which wait for multiple RPCs.
  """
    __method = None
    __get_result_hook = None
    __user_data = None
    __postcall_hooks_called = False
    __must_call_user_callback = False

    class MyLocal(threading.local):
        """Class to hold per-thread class level attributes."""
        may_interrupt_wait = False
    __local = MyLocal()

    def __init__(self, service, deadline=None, callback=None, stubmap=None):
        """Constructor.

    Args:
      service: The service name.
      deadline: Optional deadline.  Default depends on the implementation.
      callback: Optional argument-less callback function.
      stubmap: optional APIProxyStubMap instance, for dependency injection.
    """
        if stubmap is None:
            stubmap = apiproxy
        self.__stubmap = stubmap
        self.__service = service
        self.__rpc = CreateRPC(service, stubmap)
        self.__rpc.deadline = deadline
        self.__rpc.callback = self.__internal_callback
        self.callback = callback
        self.__class__.__local.may_interrupt_wait = False

    def __internal_callback(self):
        """This is the callback set on the low-level RPC object.

    It sets a flag on the current object indicating that the high-level
    callback should now be called.  If interrupts are enabled, it also
    interrupts the current wait_any() call by raising an exception.
    """
        self.__must_call_user_callback = True
        self.__rpc.callback = None
        if self.__class__.__local.may_interrupt_wait and (not self.__rpc.exception):
            raise apiproxy_errors.InterruptedError(None, self.__rpc)

    @property
    def service(self):
        """Return the service name."""
        return self.__service

    @property
    def method(self):
        """Return the method name."""
        return self.__method

    @property
    def deadline(self):
        """Return the deadline, if set explicitly (otherwise None)."""
        return self.__rpc.deadline

    @property
    def request(self):
        """Return the request protocol buffer object."""
        return self.__rpc.request

    @property
    def response(self):
        """Return the response protocol buffer object."""
        return self.__rpc.response

    @property
    def state(self):
        """Return the RPC state.

    Possible values are attributes of apiproxy_rpc.RPC: IDLE, RUNNING,
    FINISHING.
    """
        return self.__rpc.state

    @property
    def get_result_hook(self):
        """Return the get-result hook function."""
        return self.__get_result_hook

    @property
    def user_data(self):
        """Return the user data for the hook function."""
        return self.__user_data

    def make_call(self, method, request, response, get_result_hook=None, user_data=None):
        """Initiate a call.

    Args:
      method: The method name.
      request: The request protocol buffer.
      response: The response protocol buffer.
      get_result_hook: Optional get-result hook function.  If not None,
        this must be a function with exactly one argument, the RPC
        object (self).  Its return value is returned from get_result().
      user_data: Optional additional arbitrary data for the get-result
        hook function.  This can be accessed as rpc.user_data.  The
        type of this value is up to the service module.

    This function may only be called once per RPC object.  It sends
    the request to the remote server, but does not wait for a
    response.  This allows concurrent execution of the remote call and
    further local processing (e.g., making additional remote calls).

    Before the call is initiated, the precall hooks are called.
    """
        assert self.__rpc.state == apiproxy_rpc.RPC.IDLE, repr(self.state)
        self.__method = method
        self.__get_result_hook = get_result_hook
        self.__user_data = user_data
        self.__stubmap.GetPreCallHooks().Call(self.__service, method, request, response, self.__rpc)
        self.__rpc.MakeCall(self.__service, method, request, response)

    def wait(self):
        """Wait for the call to complete, and call callback if needed.

    This and wait_any()/wait_all() are the only time callback
    functions may be called.  (However, note that check_success() and
    get_result() call wait().)  Waiting for one RPC will not cause
    callbacks for other RPCs to be called.  Callback functions may
    call check_success() and get_result().

    Callbacks are called without arguments; if a callback needs access
    to the RPC object a Python nested function (a.k.a. closure) or a
    bound may be used.  To facilitate this, the callback may be
    assigned after the RPC object is created (but before make_call()
    is called).

    Note: don't confuse callbacks with get-result hooks or precall
    and postcall hooks.
    """
        assert self.__rpc.state != apiproxy_rpc.RPC.IDLE, repr(self.state)
        if self.__rpc.state == apiproxy_rpc.RPC.RUNNING:
            self.__rpc.Wait()
        assert self.__rpc.state == apiproxy_rpc.RPC.FINISHING, repr(self.state)
        self.__call_user_callback()

    def __call_user_callback(self):
        """Call the high-level callback, if requested."""
        if self.__must_call_user_callback:
            self.__must_call_user_callback = False
            if self.callback is not None:
                self.callback()

    def check_success(self):
        """Check for success of the RPC, possibly raising an exception.

    This function should be called at least once per RPC.  If wait()
    hasn't been called yet, it is called first.  If the RPC caused
    an exceptional condition, an exception will be raised here.
    The first time check_success() is called, the postcall hooks
    are called.
    """
        self.wait()
        try:
            self.__rpc.CheckSuccess()
        except Exception as err:
            if not self.__postcall_hooks_called:
                self.__postcall_hooks_called = True
                self.__stubmap.GetPostCallHooks().Call(self.__service, self.__method, self.request, self.response, self.__rpc, err)
            raise
        else:
            if not self.__postcall_hooks_called:
                self.__postcall_hooks_called = True
                self.__stubmap.GetPostCallHooks().Call(self.__service, self.__method, self.request, self.response, self.__rpc)

    def get_result(self):
        """Get the result of the RPC, or possibly raise an exception.

    This implies a call to check_success().  If a get-result hook was
    passed to make_call(), that hook is responsible for calling
    check_success(), and the return value of the hook is returned.
    Otherwise, check_success() is called directly and None is
    returned.
    """
        if self.__get_result_hook is None:
            self.check_success()
            return None
        else:
            return self.__get_result_hook(self)

    @classmethod
    def __check_one(cls, rpcs):
        """Check the list of RPCs for one that is finished, or one that is running.

    Args:
      rpcs: Iterable collection of UserRPC instances.

    Returns:
      A pair (finished, running), as follows:
      (UserRPC, None) indicating the first RPC found that is finished;
      (None, UserRPC) indicating the first RPC found that is running;
      (None, None) indicating no RPCs are finished or running.
    """
        rpc = None
        for rpc in rpcs:
            assert isinstance(rpc, cls), repr(rpc)
            state = rpc.__rpc.state
            if state == apiproxy_rpc.RPC.FINISHING:
                rpc.__call_user_callback()
                return (rpc, None)
            assert state != apiproxy_rpc.RPC.IDLE, repr(rpc)
        return (None, rpc)

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

    @classmethod
    def wait_all(cls, rpcs):
        """Wait until all given RPCs are finished.

    This is a thin wrapper around wait_any() that loops until all
    given RPCs have finished.

    Args:
      rpcs: Iterable collection of UserRPC instances.

    Returns:
      None.
    """
        rpcs = set(rpcs)
        while rpcs:
            finished = cls.wait_any(rpcs)
            if finished is not None:
                rpcs.remove(finished)