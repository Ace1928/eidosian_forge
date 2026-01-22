import collections
import contextlib
import functools
import itertools
import textwrap
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import lazy_variable
from keras.src.engine import base_layer_utils
from keras.src.engine import input_spec
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
from keras.src.mixed_precision import autocast_variable
from keras.src.mixed_precision import policy
from keras.src.saving import serialization_lib
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import generic_utils
from keras.src.utils import layer_utils
from keras.src.utils import object_identity
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.generic_utils import to_snake_case  # noqa: F401
from keras.src.utils.tf_utils import is_tensor_or_tensor_list  # noqa: F401
from google.protobuf import json_format
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import (
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
@keras_export('keras.__internal__.layers.BaseRandomLayer')
class BaseRandomLayer(Layer):
    """A layer handle the random number creation and savemodel behavior."""

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, seed=None, force_generator=False, rng_type=None, **kwargs):
        """Initialize the BaseRandomLayer.

        Note that the constructor is annotated with
        @no_automatic_dependency_tracking. This is to skip the auto
        tracking of self._random_generator instance, which is an AutoTrackable.
        The backend.RandomGenerator could contain a tf.random.Generator instance
        which will have tf.Variable as the internal state. We want to avoid
        saving that state into model.weights and checkpoints for backward
        compatibility reason. In the meantime, we still need to make them
        visible to SavedModel when it is tracing the tf.function for the
        `call()`.
        See _list_extra_dependencies_for_serialization below for more details.

        Args:
          seed: optional integer, used to create RandomGenerator.
          force_generator: boolean, default to False, whether to force the
            RandomGenerator to use the code branch of tf.random.Generator.
          rng_type: string, the rng type that will be passed to backend
            RandomGenerator. `None` will allow RandomGenerator to choose
            types by itself. Valid values are "stateful", "stateless",
            "legacy_stateful". Defaults to `None`.
          **kwargs: other keyword arguments that will be passed to the parent
            *class
        """
        super().__init__(**kwargs)
        self._random_generator = backend.RandomGenerator(seed, force_generator=force_generator, rng_type=rng_type)

    def build(self, input_shape):
        super().build(input_shape)
        self._random_generator._maybe_init()

    def _trackable_children(self, save_type='checkpoint', **kwargs):
        if save_type == 'savedmodel':
            cache = kwargs['cache']
            children = self._trackable_saved_model_saver.trackable_children(cache)
            children['_random_generator'] = self._random_generator
        else:
            children = {}
        children.update(super()._trackable_children(save_type, **kwargs))
        return children

    def _lookup_dependency(self, name, cached_dependencies=None):
        if name == '_random_generator':
            return self._random_generator
        elif cached_dependencies is not None:
            return cached_dependencies.get(name)
        else:
            return super()._lookup_dependency(name)