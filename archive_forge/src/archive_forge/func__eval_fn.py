import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def _eval_fn(strategy):
    """Function for evaluator task."""
    local_estimator = copy.deepcopy(estimator)
    local_estimator._config._eval_distribute = strategy
    _init_run_config_from_worker_context(local_estimator._config, dc_context.get_current_worker_context())
    logging.info('Updated config: %s', str(vars(local_estimator._config)))
    local_estimator._eval_distribution = strategy
    local_estimator._config._distribute_coordinator_mode = None
    executor = executor_cls(local_estimator, train_spec, eval_spec)
    executor._start_continuous_evaluation()