from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export
Reduce `value` across replicas.

    Given a per-replica value returned by `run`, say a
    per-example loss, the batch will be divided across all the replicas. This
    function allows you to aggregate across replicas and optionally also across
    batch elements.  For example, if you have a global batch size of 8 and 2
    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and
    `[4, 5, 6, 7]` will be on replica 1. By default, `reduce` will just
    aggregate across replicas, returning `[0+4, 1+5, 2+6, 3+7]`. This is useful
    when each replica is computing a scalar or some other value that doesn't
    have a "batch" dimension (like a gradient). More often you will want to
    aggregate across the global batch, which you can get by specifying the batch
    dimension as the `axis`, typically `axis=0`. In this case it would return a
    scalar `0+1+2+3+4+5+6+7`.

    If there is a last partial batch, you will need to specify an axis so
    that the resulting shape is consistent across replicas. So if the last
    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you
    would get a shape mismatch unless you specify `axis=0`. If you specify
    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct
    denominator of 6. Contrast this with computing `reduce_mean` to get a
    scalar value on each replica and this function to average those means,
    which will weigh some values `1/8` and others `1/4`.

    For Example:
    ```
    strategy = tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=['CPU:0', 'GPU:0'], parameter_device='CPU:0')
    ds = tf.data.Dataset.range(10)
    # Distribute that dataset
    dist_dataset = strategy.experimental_distribute_dataset(ds)

    with strategy.scope():
      @tf.function
      def train_step(val):
        # pass through
        return val

      # Iterate over the distributed dataset
      for x in dist_dataset:
        result = strategy.run(train_step, args=(x,))

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result,
                             axis=None).numpy()
    # result: array([ 4,  6,  8, 10])

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result, axis=0).numpy()
    # result: 28
    ```

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: A "per replica" value, e.g. returned by `run` to
        be combined into a single tensor.
      axis: Specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    