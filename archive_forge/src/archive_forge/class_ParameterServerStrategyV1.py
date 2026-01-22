import copy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distribute.experimental.ParameterServerStrategy'])
class ParameterServerStrategyV1(distribute_lib.StrategyV1):
    """An asynchronous multi-worker parameter server tf.distribute strategy.

  This strategy requires two roles: workers and parameter servers. Variables and
  updates to those variables will be assigned to parameter servers and other
  operations are assigned to workers.

  When each worker has more than one GPU, operations will be replicated on all
  GPUs. Even though operations may be replicated, variables are not and each
  worker shares a common view for which parameter server a variable is assigned
  to.

  By default it uses `TFConfigClusterResolver` to detect configurations for
  multi-worker training. This requires a 'TF_CONFIG' environment variable and
  the 'TF_CONFIG' must have a cluster spec.

  This class assumes each worker is running the same code independently, but
  parameter servers are running a standard server. This means that while each
  worker will synchronously compute a single gradient update across all GPUs,
  updates between workers proceed asynchronously. Operations that occur only on
  the first replica (such as incrementing the global step), will occur on the
  first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.compat.v1.colocate_with`) under the strategy's scope. For colocating
  variables, use `strategy.extended.colocate_vars_with` instead. Colocation of
  ops will possibly create device assignment conflicts.

  Note: This strategy only works with the Estimator API. Pass an instance of
  this strategy to the `experimental_distribute` argument when you create the
  `RunConfig`. This instance of `RunConfig` should then be passed to the
  `Estimator` instance on which `train_and_evaluate` is called.

  For Example:
  ```
  strategy = tf.distribute.experimental.ParameterServerStrategy()
  run_config = tf.estimator.RunConfig(
      experimental_distribute.train_distribute=strategy)
  estimator = tf.estimator.Estimator(config=run_config)
  tf.estimator.train_and_evaluate(estimator,...)
  ```
  """

    def __init__(self, cluster_resolver=None):
        """Initializes this strategy with an optional `cluster_resolver`.

    Args:
      cluster_resolver: Optional
        `tf.distribute.cluster_resolver.ClusterResolver` object. Defaults to a
        `tf.distribute.cluster_resolver.TFConfigClusterResolver`.
    """
        if cluster_resolver is None:
            cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        super(ParameterServerStrategyV1, self).__init__(ParameterServerStrategyExtended(self, cluster_resolver=cluster_resolver))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('ParameterServerStrategy')

    def experimental_distribute_dataset(self, dataset, options=None):
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function`.')
        self._raise_pss_error_if_eager()
        super(ParameterServerStrategyV1, self).experimental_distribute_dataset(dataset=dataset, options=options)

    def distribute_datasets_from_function(self, dataset_fn, options=None):
        if options and options.experimental_replication_mode == distribute_lib.InputReplicationMode.PER_REPLICA:
            raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function` of tf.distribute.MirroredStrategy')
        self._raise_pss_error_if_eager()
        super(ParameterServerStrategyV1, self).distribute_datasets_from_function(dataset_fn=dataset_fn, options=options)

    def run(self, fn, args=(), kwargs=None, options=None):
        self._raise_pss_error_if_eager()
        super(ParameterServerStrategyV1, self).run(fn, args=args, kwargs=kwargs, options=options)

    def scope(self):
        self._raise_pss_error_if_eager()
        return super(ParameterServerStrategyV1, self).scope()

    def _raise_pss_error_if_eager(self):
        if context.executing_eagerly():
            raise NotImplementedError('`tf.compat.v1.distribute.experimental.ParameterServerStrategy` currently only works with the tf.Estimator API')