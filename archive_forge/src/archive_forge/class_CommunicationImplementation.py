import copy
import enum
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.CommunicationImplementation', 'distribute.experimental.CollectiveCommunication')
class CommunicationImplementation(enum.Enum):
    """Cross device communication implementation.

  Warning: The alias `tf.distribute.experimental.CollectiveCommunication` is
  deprecated and will be removed in a future version. Use
  `tf.distribute.experimental.CommunicationImplementation` instead.

  * `AUTO`: Automatically chosen by Tensorflow.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: NVIDIA®'s NCCL library. This is now only used for all-reduce on
    GPUs; all-reduce on CPU, all-gather and broadcast fallbacks to RING.
  """
    AUTO = 'AUTO'
    RING = 'RING'
    NCCL = 'NCCL'