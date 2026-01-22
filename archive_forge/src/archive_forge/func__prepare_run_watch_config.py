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