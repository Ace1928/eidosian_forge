from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distribute.OneDeviceStrategy'])
class OneDeviceStrategyV1(distribute_lib.StrategyV1):
    __doc__ = OneDeviceStrategy.__doc__.replace('For example:\n  ```', 'For example:\n  ```\n  tf.enable_eager_execution()')

    def __init__(self, device):
        super(OneDeviceStrategyV1, self).__init__(OneDeviceExtended(self, device))
        distribute_lib.distribution_strategy_gauge.get_cell('V1').set('OneDeviceStrategy')
    __init__.__doc__ = OneDeviceStrategy.__init__.__doc__