import collections
import copy
import re
import sys
import types
import unittest
from absl import app
import six
from tensorflow.python.client import session
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _multi_worker_session(kwargs):
    """Returns a context manager that enters a session that is configured for the MultiWorkerMirroredStrategy.

  Args:
    kwargs: a dict. Keyword arguments passed to the test.

  Returns:
    A context manager. If MultiWorkerMirroredStrategy is the  one and only one
    strategy in kwargs and it's in graph mode, it's the seesion that is
    configured for that strategy.  Otherwise, it's a no-op context manager.
  """
    strategy = None
    for _, v in kwargs.items():
        if isinstance(v, distribute_lib.StrategyBase):
            if strategy is not None:
                logging.warning('The test uses multiple strategies. Skipping entering a session that is configured for the strategy.')
                return ops.NullContextmanager()
            strategy = v
    if context.executing_eagerly() or not isinstance(strategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy):
        return ops.NullContextmanager()
    sess_config = copy.deepcopy(context.context().config)
    sess_config = strategy.update_config_proto(sess_config)
    target = strategy.cluster_resolver.master()
    return session.Session(config=sess_config, target=target).as_default()