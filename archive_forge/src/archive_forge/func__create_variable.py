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
def _create_variable(self, next_creator, **kwargs):
    kwargs.pop('use_resource', None)
    kwargs.pop('colocate_with', None)
    kwargs.pop('expected_shape', None)
    initial_value = kwargs.pop('initial_value')
    dtype = kwargs.get('dtype', None)

    def new_initial_value():
        if callable(initial_value):
            init_var = ops.convert_to_tensor(initial_value(), dtype=dtype)
        else:
            init_var = ops.convert_to_tensor(initial_value, dtype=dtype)
        rank = init_var.shape.rank
        return d_api.copy_to_mesh(init_var, layout.Layout.replicated(self._mesh, rank))
    return d_variable.DVariable(new_initial_value, **kwargs)