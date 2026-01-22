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
def _run_with_profiling(self, run_start_resp, fetches, feed_dict, options, run_metadata, callable_runner, callable_runner_args, callable_options):
    """Perform a session.run() or callable with profiling."""
    decorated_run_options = None
    if callable_options:
        callable_options_id = id(callable_options)
        if callable_options_id not in self._cached_callables_from_options:
            new_callable_options = config_pb2.CallableOptions()
            new_callable_options.CopyFrom(callable_options)
            decorated_run_options = new_callable_options.run_options
    else:
        decorated_run_options = options or config_pb2.RunOptions()
    self._decorate_run_options_for_profile(decorated_run_options)
    run_metadata = run_metadata or config_pb2.RunMetadata()
    if callable_runner:
        retvals = callable_runner(*callable_runner_args, options=decorated_run_options, run_metadata=run_metadata)
    elif callable_options:
        callable_object = self._sess._make_callable_from_options(new_callable_options)
        retvals = callable_object(*callable_runner_args, run_metadata=run_metadata)
    else:
        retvals = self._sess.run(fetches, feed_dict=feed_dict, options=decorated_run_options, run_metadata=run_metadata)
    return (retvals, OnRunEndRequest(run_start_resp.action, run_metadata=run_metadata, client_graph_def=self._sess.graph.as_graph_def()))