import collections
import itertools
import json
import os
import random
import sys
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.distribute import distribute_coordinator_utils as dc
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.engine import keras_tensor
from keras.src.utils import control_flow_util
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
class RandomGenerator(tf.__internal__.tracking.AutoTrackable):
    """Random generator that selects appropriate random ops.

    This class contains the logic for legacy stateful random ops, as well as the
    new stateless random ops with seeds and tf.random.Generator. Any class that
    relies on RNG (eg initializer, shuffle, dropout) should use this class to
    handle the transition from legacy RNGs to new RNGs.

    Args:
      seed: Optional int seed. When `rng_type` is "stateful", the seed is used
        to create `tf.random.Generator` to produce deterministic sequences.
        When `rng_type` is "stateless", new seed will be created if it is not
        provided by user, and it will be passed down to stateless random ops.
        When `rng_type` is "legacy_stateful", the seed will be passed down to
        stateful random ops.
      rng_type: Type of RNG to use, one of "stateful", "stateless",
        "legacy_stateful". When `None` it uses "stateful" if
        `enable_tf_random_generator` has been activated, or
        "legacy_stateful" otherwise.
        - When using "stateless", the random ops outputs are constant (the same
          inputs result in the same outputs).
        - When using "stateful" or "legacy_stateful", the random ops outputs are
          non-constant, but deterministic: calling the same random op multiple
          times with the same inputs results in a deterministic sequence of
          different outputs.
        - "legacy_stateful" is backed by TF1 stateful RNG ops
          (e.g. `tf.random.uniform`), while "stateful"
          is backed by TF2 APIs (e.g. `tf.random.Generator.uniform`).
        Defaults to `None`.
    """
    RNG_STATELESS = 'stateless'
    RNG_STATEFUL = 'stateful'
    RNG_LEGACY_STATEFUL = 'legacy_stateful'

    def __init__(self, seed=None, rng_type=None, **kwargs):
        self._seed = seed
        self._set_rng_type(rng_type, **kwargs)
        self._built = False

    def _set_rng_type(self, rng_type, **kwargs):
        if kwargs.get('force_generator', False):
            rng_type = self.RNG_STATEFUL
        if rng_type is None:
            if is_tf_random_generator_enabled():
                self._rng_type = self.RNG_STATEFUL
            else:
                self._rng_type = self.RNG_LEGACY_STATEFUL
        else:
            if rng_type not in [self.RNG_STATEFUL, self.RNG_LEGACY_STATEFUL, self.RNG_STATELESS]:
                raise ValueError(f'Invalid `rng_type` received. Valid `rng_type` are ["stateless", "stateful", "legacy_stateful"]. Got: {rng_type}')
            self._rng_type = rng_type

    def _maybe_init(self):
        """Lazily init the RandomGenerator.

        The TF API executing_eagerly_outside_functions() has some side effect,
        and couldn't be used before API like tf.enable_eager_execution(). Some
        of the client side code was creating the initializer at the code load
        time, which triggers the creation of RandomGenerator. Lazy init this
        class to walkaround this issue until it is resolved on TF side.
        """
        if self._built:
            return
        if self._rng_type == self.RNG_STATEFUL and (not tf.compat.v1.executing_eagerly_outside_functions()):
            self._rng_type = self.RNG_LEGACY_STATEFUL
        if self._rng_type == self.RNG_STATELESS:
            self._seed = self._create_seed(self._seed)
            self._generator = None
        elif self._rng_type == self.RNG_STATEFUL:
            with tf_utils.maybe_init_scope(self):
                seed = self._create_seed(self._seed)
                self._generator = tf.random.Generator.from_seed(seed, alg=tf.random.Algorithm.AUTO_SELECT)
        else:
            self._generator = None
        self._built = True

    def make_seed_for_stateless_op(self):
        """Generate a new seed based on the init config.

        Note that this will not return python ints which will be frozen in the
        graph and cause stateless op to return the same value. It will only
        return value when generator is used, otherwise it will return None.

        Returns:
          A tensor with shape [2,].
        """
        self._maybe_init()
        if self._rng_type == self.RNG_STATELESS:
            return [self._seed, 0]
        elif self._rng_type == self.RNG_STATEFUL:
            return self._generator.make_seeds()[:, 0]
        return None

    def make_legacy_seed(self):
        """Create a new seed for the legacy stateful ops to use.

        When user didn't provide any original seed, this method will return
        None.  Otherwise it will increment the counter and return as the new
        seed.

        Note that it is important to generate different seed for stateful ops in
        the `tf.function`. The random ops will return same value when same seed
        is provided in the `tf.function`.

        Returns:
          int as new seed, or None.
        """
        if self._seed is not None:
            result = self._seed
            self._seed += 1
            return result
        return None

    def _create_seed(self, user_specified_seed):
        if user_specified_seed is not None:
            return user_specified_seed
        elif getattr(_SEED_GENERATOR, 'generator', None):
            return _SEED_GENERATOR.generator.randint(1, 1000000000.0)
        else:
            return random.randint(1, int(1000000000.0))

    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, nonce=None):
        """Produce random number based on the normal distribution.

        Args:
          shape: The shape of the random values to generate.
          mean: Floats, default to 0. Mean of the random values to generate.
          stddev: Floats, default to 1. Standard deviation of the random values
            to generate.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype)
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)
            return tf.random.stateless_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
        return tf.random.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.make_legacy_seed())

    def random_uniform(self, shape, minval=0.0, maxval=None, dtype=None, nonce=None):
        """Produce random number based on the uniform distribution.

        Args:
          shape: The shape of the random values to generate.
          minval: Floats, default to 0. Lower bound of the range of
            random values to generate (inclusive).
          minval: Floats, default to None. Upper bound of the range of
            random values to generate (exclusive).
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)
            return tf.random.stateless_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
        return tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=self.make_legacy_seed())

    def truncated_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, nonce=None):
        """Produce random number based on the truncated normal distribution.

        Args:
          shape: The shape of the random values to generate.
          mean: Floats, default to 0. Mean of the random values to generate.
          stddev: Floats, default to 1. Standard deviation of the random values
            to generate.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype)
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)
            return tf.random.stateless_truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
        return tf.random.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.make_legacy_seed())

    def dropout(self, inputs, rate, noise_shape=None):
        self._maybe_init()
        if self._rng_type == self.RNG_STATEFUL:
            return tf.nn.experimental.general_dropout(inputs, rate=rate, noise_shape=noise_shape, uniform_sampler=self._generator.uniform)
        elif self._rng_type == self.RNG_STATELESS:
            return tf.nn.experimental.stateless_dropout(inputs, rate=rate, noise_shape=noise_shape, seed=self.make_seed_for_stateless_op())
        else:
            return tf.nn.dropout(inputs, rate=rate, noise_shape=noise_shape, seed=self.make_legacy_seed())