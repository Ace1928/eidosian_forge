import functools
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def _gather_to_implementation(self, value, destinations, axis, options):
    if isinstance(value, dtensor_util.DTensorDistributedValue):
        value = value.get_dtensor()
    if not d_api.is_dtensor(value):
        return value
    components = d_api.unpack(value)
    return array_ops.concat(components, axis=axis)