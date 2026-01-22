import inspect
import itertools
import string
from absl import logging
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.layers import Layer
from keras.src.models import Functional
from keras.src.models import Sequential
from keras.src.utils import io_utils
from keras.src.utils import tree
from keras.src.utils.module_utils import tensorflow as tf
def _convert_jax2tf_function(self, fn, input_signature, jax2tf_kwargs=None):
    from jax.experimental import jax2tf
    if jax2tf_kwargs is None:
        jax2tf_kwargs = {}
    if 'native_serialization' not in jax2tf_kwargs:
        jax2tf_kwargs['native_serialization'] = self._check_device_compatible()
    variables_shapes = self._to_polymorphic_shape(self._backend_variables, allow_none=False)
    if 'polymorphic_shapes' in jax2tf_kwargs:
        input_shapes = jax2tf_kwargs['polymorphic_shapes']
    else:
        input_shapes = self._to_polymorphic_shape(input_signature)
    jax2tf_kwargs['polymorphic_shapes'] = [variables_shapes] + input_shapes
    return jax2tf.convert(fn, **jax2tf_kwargs)