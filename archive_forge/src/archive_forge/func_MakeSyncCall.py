from __future__ import absolute_import
import inspect
import sys
import threading
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def MakeSyncCall(self, service, call, request, response):
    """The APIProxy entry point.

    Args:
      service: string representing which service to call
      call: string representing which function to call
      request: protocol buffer for the request
      response: protocol buffer for the response

    Returns:
      Response protocol buffer or None. Some implementations may return
      a response protocol buffer instead of modifying 'response'.
      Caller must use returned value in such cases. If 'response' is modified
      then returns None.

    Raises:
      apiproxy_errors.Error or a subclass.
    """
    stub = self.GetStub(service)
    assert stub, 'No api proxy found for service "%s"' % service
    if hasattr(stub, 'CreateRPC'):
        rpc = stub.CreateRPC()
        self.__precall_hooks.Call(service, call, request, response, rpc)
        try:
            rpc.MakeCall(service, call, request, response)
            rpc.Wait()
            rpc.CheckSuccess()
        except Exception as err:
            self.__postcall_hooks.Call(service, call, request, response, rpc, err)
            raise
        else:
            self.__postcall_hooks.Call(service, call, request, response, rpc)
    else:
        self.__precall_hooks.Call(service, call, request, response)
        try:
            returned_response = stub.MakeSyncCall(service, call, request, response)
        except Exception as err:
            self.__postcall_hooks.Call(service, call, request, response, None, err)
            raise
        else:
            self.__postcall_hooks.Call(service, call, request, returned_response or response)
            return returned_response