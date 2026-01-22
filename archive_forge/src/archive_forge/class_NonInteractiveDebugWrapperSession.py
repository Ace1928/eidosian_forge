import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class NonInteractiveDebugWrapperSession(BaseDebugWrapperSession):
    """Base class for non-interactive (i.e., non-CLI) debug wrapper sessions."""

    def __init__(self, sess, watch_fn=None, thread_name_filter=None, pass_through_operrors=False):
        """Constructor of NonInteractiveDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      watch_fn: (`Callable`) A Callable that maps the fetches and feeds of a
        debugged `Session.run()` call to `WatchOptions.`
        * Args:
          * `fetches`: the fetches to the `Session.run()` call.
          * `feeds`: the feeds to the `Session.run()` call.

        * Returns:
         (`tf_debug.WatchOptions`) An object containing debug options including
           the debug ops to use, the node names, op types and/or tensor data
           types to watch, etc. See the documentation of `tf_debug.WatchOptions`
           for more details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      pass_through_operrors: If true, all captured OpErrors will be
        propagated.  By default this captures all OpErrors.
    Raises:
       TypeError: If a non-None `watch_fn` is specified and it is not callable.
    """
        BaseDebugWrapperSession.__init__(self, sess, thread_name_filter=thread_name_filter, pass_through_operrors=pass_through_operrors)
        self._watch_fn = None
        if watch_fn is not None:
            if not callable(watch_fn):
                raise TypeError('watch_fn is not callable')
            self._watch_fn = watch_fn

    def on_session_init(self, request):
        """See doc of BaseDebugWrapperSession.on_run_start."""
        return OnSessionInitResponse(OnSessionInitAction.PROCEED)

    @abc.abstractmethod
    def prepare_run_debug_urls(self, fetches, feed_dict):
        """Abstract method to be implemented by concrete subclasses.

    This method prepares the run-specific debug URL(s).

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) Debug URLs to be used in
        this `Session.run()` call.
    """

    def on_run_start(self, request):
        """See doc of BaseDebugWrapperSession.on_run_start."""
        debug_urls, watch_opts = self._prepare_run_watch_config(request.fetches, request.feed_dict)
        return OnRunStartResponse(OnRunStartAction.DEBUG_RUN, debug_urls, debug_ops=watch_opts.debug_ops, node_name_regex_allowlist=watch_opts.node_name_regex_allowlist, op_type_regex_allowlist=watch_opts.op_type_regex_allowlist, tensor_dtype_regex_allowlist=watch_opts.tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=watch_opts.tolerate_debug_op_creation_failures)

    def _prepare_run_watch_config(self, fetches, feed_dict):
        """Get the debug_urls, and node/op allowlists for the current run() call.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`.
      feed_dict: Same as the `feed_dict argument` to `Session.run()`.

    Returns:
      debug_urls: (str or list of str) Debug URLs for the current run() call.
        Currently, the list consists of only one URL that is a file:// URL.
      watch_options: (WatchOptions) The return value of a watch_fn, containing
        options including debug_ops, and allowlists.
    """
        debug_urls = self.prepare_run_debug_urls(fetches, feed_dict)
        if self._watch_fn is None:
            watch_options = WatchOptions()
        else:
            watch_options = self._watch_fn(fetches, feed_dict)
            if isinstance(watch_options, tuple):
                watch_options = WatchOptions(*watch_options)
        return (debug_urls, watch_options)

    def on_run_end(self, request):
        """See doc of BaseDebugWrapperSession.on_run_end."""
        return OnRunEndResponse()