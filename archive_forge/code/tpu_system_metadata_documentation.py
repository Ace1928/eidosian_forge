import collections
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu
from tensorflow.python.util.tf_export import tf_export
Returns the canonical job name to use to place TPU computations on.

  Args:
    master: A `string` representing the TensorFlow master to use.
    cluster_def: A ClusterDef object describing the TPU cluster.

  Returns:
    A string containing the job name, or None if no job should be specified.

  Raises:
    ValueError: If the user needs to specify a tpu_job_name, because we are
      unable to infer the job name automatically, or if the user-specified job
      names are inappropriate.
  