import functools
import numpy as np
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _prepare_feed_values(model, inputs, targets, sample_weights, mode):
    """Prepare feed values to the model execution function.

  Args:
    model: Model to prepare feed values for.
    inputs: List or dict of model inputs.
    targets: Optional list of model targets.
    sample_weights: Optional list of sample weight arrays.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.

  Returns:
    Feed values for the model in the given mode.
  """
    if model._distribution_strategy:
        if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2)):
            inputs = distributed_training_utils_v1.get_iterator(inputs, model._distribution_strategy)

        def get_distributed_inputs():
            return distributed_training_utils_v1._prepare_feed_values(model, inputs, targets, sample_weights, mode)
        if context.executing_eagerly():
            return get_distributed_inputs
        else:
            return get_distributed_inputs()
    if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator)):
        inputs, targets, sample_weights = model._standardize_user_data(inputs, extract_tensors_from_dataset=True)
    inputs = training_utils_v1.ModelInputs(inputs).as_list()
    targets = list(targets or [])
    sample_weights = list(sample_weights or [])
    ins = inputs + targets + sample_weights
    if mode == ModeKeys.TRAIN and (not isinstance(backend.symbolic_learning_phase(), int)):
        ins += [True]
    return ins