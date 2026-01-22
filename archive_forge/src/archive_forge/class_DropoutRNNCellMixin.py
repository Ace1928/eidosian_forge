import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
@doc_controls.do_not_generate_docs
class DropoutRNNCellMixin(object):
    """Object that hold dropout related fields for RNN Cell.

  This class is not a standalone RNN cell. It suppose to be used with a RNN cell
  by multiple inheritance. Any cell that mix with class should have following
  fields:
    dropout: a float number within range [0, 1). The ratio that the input
      tensor need to dropout.
    recurrent_dropout: a float number within range [0, 1). The ratio that the
      recurrent state weights need to dropout.
  This object will create and cache created dropout masks, and reuse them for
  the incoming data, so that the same mask is used for every batch input.
  """

    def __init__(self, *args, **kwargs):
        self._create_non_trackable_mask_cache()
        super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

    @trackable.no_automatic_dependency_tracking
    def _create_non_trackable_mask_cache(self):
        """Create the cache for dropout and recurrent dropout mask.

    Note that the following two masks will be used in "graph function" mode,
    e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
    tensors will be generated differently than in the "graph function" case,
    and they will be cached.

    Also note that in graph mode, we still cache those masks only because the
    RNN could be created with `unroll=True`. In that case, the `cell.call()`
    function will be invoked multiple times, and we want to ensure same mask
    is used every time.

    Also the caches are created without tracking. Since they are not picklable
    by python when deepcopy, we don't want `layer._obj_reference_counts_dict`
    to track it by default.
    """
        self._dropout_mask_cache = backend.ContextValueCache(self._create_dropout_mask)
        self._recurrent_dropout_mask_cache = backend.ContextValueCache(self._create_recurrent_dropout_mask)

    def reset_dropout_mask(self):
        """Reset the cached dropout masks if any.

    This is important for the RNN layer to invoke this in it `call()` method so
    that the cached mask is cleared before calling the `cell.call()`. The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
        self._dropout_mask_cache.clear()

    def reset_recurrent_dropout_mask(self):
        """Reset the cached recurrent dropout masks if any.

    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
        self._recurrent_dropout_mask_cache.clear()

    def _create_dropout_mask(self, inputs, training, count=1):
        return _generate_dropout_mask(array_ops.ones_like(inputs), self.dropout, training=training, count=count)

    def _create_recurrent_dropout_mask(self, inputs, training, count=1):
        return _generate_dropout_mask(array_ops.ones_like(inputs), self.recurrent_dropout, training=training, count=count)

    def get_dropout_mask_for_cell(self, inputs, training, count=1):
        """Get the dropout mask for RNN cell's input.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
        if self.dropout == 0:
            return None
        init_kwargs = dict(inputs=inputs, training=training, count=count)
        return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)

    def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
        """Get the recurrent dropout mask for RNN cell.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
        if self.recurrent_dropout == 0:
            return None
        init_kwargs = dict(inputs=inputs, training=training, count=count)
        return self._recurrent_dropout_mask_cache.setdefault(kwargs=init_kwargs)

    def __getstate__(self):
        state = super(DropoutRNNCellMixin, self).__getstate__()
        state.pop('_dropout_mask_cache', None)
        state.pop('_recurrent_dropout_mask_cache', None)
        return state

    def __setstate__(self, state):
        state['_dropout_mask_cache'] = backend.ContextValueCache(self._create_dropout_mask)
        state['_recurrent_dropout_mask_cache'] = backend.ContextValueCache(self._create_recurrent_dropout_mask)
        super(DropoutRNNCellMixin, self).__setstate__(state)