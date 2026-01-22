import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
class CombinerPreprocessingLayer(PreprocessingLayer):
    """Base class for PreprocessingLayers that do computation using a Combiner.

  This class provides several helper methods to make creating a
  PreprocessingLayer easier. It assumes that the core of your computation will
  be done via a Combiner object. Subclassing this class to create a
  PreprocessingLayer allows your layer to be compatible with distributed
  computation.

  This class is compatible with Tensorflow 2.0+.
  """

    def __init__(self, combiner, **kwargs):
        super(CombinerPreprocessingLayer, self).__init__(**kwargs)
        self.state_variables = collections.OrderedDict()
        self._combiner = combiner
        self._adapt_accumulator = None

    def reset_state(self):
        self._adapt_accumulator = None

    @trackable.no_automatic_dependency_tracking
    def update_state(self, data):
        if self._adapt_accumulator is None:
            self._adapt_accumulator = self._get_accumulator()
        self._adapt_accumulator = self._combiner.compute(data, self._adapt_accumulator)

    def merge_state(self, layers):
        accumulators = [self._get_accumulator()] + [l._get_accumulator() for l in layers]
        merged_accumulator = self._combiner.merge(accumulators)
        self._set_accumulator(merged_accumulator)

    def finalize_state(self):
        if self._adapt_accumulator is not None:
            self._set_accumulator(self._adapt_accumulator)

    def compile(self, run_eagerly=None, steps_per_execution=None):
        if run_eagerly is None:
            run_eagerly = True
        super(CombinerPreprocessingLayer, self).compile(run_eagerly=run_eagerly, steps_per_execution=steps_per_execution)

    def adapt(self, data, batch_size=None, steps=None, reset_state=True):
        if not reset_state:
            self._adapt_accumulator = self._combiner.restore(self._restore_updates())
        super(CombinerPreprocessingLayer, self).adapt(data, batch_size=batch_size, steps=steps, reset_state=reset_state)

    def _add_state_variable(self, name, shape, dtype, initializer=None, partitioner=None, use_resource=None, **kwargs):
        """Add a variable that can hold state which is updated during adapt().

    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      partitioner: Partitioner to be passed to the `Trackable` API.
      use_resource: Whether to use `ResourceVariable`
      **kwargs: Additional keyword arguments. Accepted values are `getter` and
        `collections`.

    Returns:
      The created variable.
    """
        weight = self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=None, trainable=False, constraint=None, partitioner=partitioner, use_resource=use_resource, **kwargs)
        self.state_variables[name] = weight
        return weight

    def _restore_updates(self):
        """Recreates a dict of updates from the layer's weights."""
        data_dict = {}
        for name, var in self.state_variables.items():
            data_dict[name] = var.numpy()
        return data_dict

    def _get_accumulator(self):
        if self._is_adapted:
            return self._combiner.restore(self._restore_updates())
        else:
            return None

    def _set_accumulator(self, accumulator):
        updates = self._combiner.extract(accumulator)
        self._set_state_variables(updates)
        self._adapt_accumulator = None

    def _set_state_variables(self, updates):
        """Directly update the internal state of this Layer.

    This method expects a string-keyed dict of {state_variable_name: state}. The
    precise nature of the state, and the names associated, are describe by
    the subclasses of CombinerPreprocessingLayer.

    Args:
      updates: A string keyed dict of weights to update.

    Raises:
      RuntimeError: if 'build()' was not called before 'set_processing_state'.
    """
        if not self.built:
            raise RuntimeError('_set_state_variables() must be called after build().')
        with ops.init_scope():
            for var_name, value in updates.items():
                self.state_variables[var_name].assign(value)