import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import io_utils
def _get_variables_used_by_endpoints(self):
    fns = [self._get_concrete_fn(name) for name in self._endpoint_names]
    return _list_variables_used_by_fns(fns)