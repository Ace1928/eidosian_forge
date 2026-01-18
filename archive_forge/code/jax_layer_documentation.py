import inspect
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import jax_utils
from keras.src.utils import tracking
from keras.src.utils import tree
from keras.src.utils.module_utils import jax

        Returns a JAX `PRNGKey` or structure of `PRNGKey`s to pass to `call_fn`.

        By default, this returns a single `PRNGKey` retrieved by calling
        `self.seed_generator.next()` when `training` is `True`, and `None` when
        `training` is `False`. Override this to return a different structure or
        to pass RNGs in inference mode too.

        Returns:
            a JAX `PRNGKey` or structure of `PRNGKey`s that will be passed as
            the `rng` argument of `call_fn`.
        