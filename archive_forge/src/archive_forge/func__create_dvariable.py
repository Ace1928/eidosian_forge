import collections
import contextlib
import re
import threading
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import lazy_variable
from keras.src.dtensor import utils
from keras.src.engine import base_layer
from tensorflow.python.util.tf_export import keras_export
def _create_dvariable(layout_map, object_path, variable):
    """Create a new variable instead of using the LazyInitVariable.

    We choose to do this since even the LazyInitVariable might behavior like
    a normal tf.Variable/DVariable, it is not future proof for any new changes
    to variable class. It will also fail the instance type check in python,
    which could affect user's code when they do any filtering based on type to
    find any variables.

    Args:
      layout_map: a LayoutMap which contains the variable_object_path (string)
        -> Layout.
      object_path: string, the object attribute path for the variable.
      variable: LazyInitVariable which will be replaced by the newly created
        tf.Variable.
    Returns:
      A new tf.Variable with correct layout information.
    """
    layout = layout_map[object_path]
    if layout is None:
        variable_rank = variable.shape.rank
        layout = dtensor.Layout.replicated(mesh=layout_map.get_default_mesh(), rank=variable_rank)
    init_val = variable._initial_value
    if callable(init_val):
        with lazy_variable.disable_init_variable_creator():
            init_val = utils.call_with_layout(init_val, layout)
    else:
        init_val = dtensor.copy_to_mesh(init_val, layout)
    variable_name = variable.name
    if variable_name.endswith(':0'):
        variable_name = variable_name[:-2]
    new_variable = dtensor.DVariable(init_val, trainable=variable.trainable, name=variable_name)
    return new_variable