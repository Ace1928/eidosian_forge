import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
class GrpcDebugWrapperSession(framework.NonInteractiveDebugWrapperSession):
    """Debug Session wrapper that send debug data to gRPC stream(s)."""

    def __init__(self, sess, grpc_debug_server_addresses, watch_fn=None, thread_name_filter=None):
        """Constructor of DumpingDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      grpc_debug_server_addresses: (`str` or `list` of `str`) Single or a list
        of the gRPC debug server addresses, in the format of
        <host:port>, with or without the "grpc://" prefix. For example:
          "localhost:7000",
          ["localhost:7000", "192.168.0.2:8000"]
      watch_fn: (`Callable`) A Callable that can be used to define per-run
        debug ops and watched tensors. See the doc of
        `NonInteractiveDebugWrapperSession.__init__()` for details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.

    Raises:
       TypeError: If `grpc_debug_server_addresses` is not a `str` or a `list`
         of `str`.
    """
        framework.NonInteractiveDebugWrapperSession.__init__(self, sess, watch_fn=watch_fn, thread_name_filter=thread_name_filter)
        if isinstance(grpc_debug_server_addresses, str):
            self._grpc_debug_server_urls = [self._normalize_grpc_url(grpc_debug_server_addresses)]
        elif isinstance(grpc_debug_server_addresses, list):
            self._grpc_debug_server_urls = []
            for address in grpc_debug_server_addresses:
                if not isinstance(address, str):
                    raise TypeError('Expected type str in list grpc_debug_server_addresses, received type %s' % type(address))
                self._grpc_debug_server_urls.append(self._normalize_grpc_url(address))
        else:
            raise TypeError('Expected type str or list in grpc_debug_server_addresses, received type %s' % type(grpc_debug_server_addresses))

    def prepare_run_debug_urls(self, fetches, feed_dict):
        """Implementation of abstract method in superclass.

    See doc of `NonInteractiveDebugWrapperSession.prepare_run_debug_urls()`
    for details.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) file:// debug URLs to be used in
        this `Session.run()` call.
    """
        return self._grpc_debug_server_urls

    def _normalize_grpc_url(self, address):
        return common.GRPC_URL_PREFIX + address if not address.startswith(common.GRPC_URL_PREFIX) else address