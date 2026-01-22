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
def _obtain_topology(master_address, cluster_def):
    """Obtains TPU fabric topology."""
    try:
        logging.info('Initializing TPU system (master: %s) to fetch topology for model parallelism. This might take a while.', master_address)
        with ops.Graph().as_default():
            session_config = get_session_config_with_timeout(_INITIAL_TPU_SYSTEM_TIMEOUT_IN_MS, cluster_def)
            with session_lib.Session(master_address, config=session_config) as sess:
                topology = sess.run(tpu.initialize_system())
                return topology
    except errors.DeadlineExceededError:
        raise ValueError('Fail to initialize TPU system with master (%s). Please double check the TPU system is functional.' % master_address)