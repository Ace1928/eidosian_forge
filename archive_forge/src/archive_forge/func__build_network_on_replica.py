import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _build_network_on_replica(model, mode, inputs=None, targets=None):
    """Build an updated model on replicas.

  We create a new Keras model while sharing the variables from the old graph.
  Building a new sub-graph is required since the original keras model creates
  placeholders for the input and the output that are not accessible till we
  call iterator.get_next() inside the step_fn for `fit`/`evaluate`/`predict`.

  The sharing of weights and layers between the old and the new model guarantee
  that we're using Strategy variables and any updates on either model are
  reflected correctly in callbacks and loop iterations.

  We need to make sure we share the optimizers between the old and the new model
  as well so that optimizer state is not lost if the user is running fit
  multiple times.

  Args:
    model: Model to be replicated across Replicas
    mode: Which of fit/eval/predict is building the distributed network
    inputs: Input variables to be passed to the model
    targets: Target tensor to be passed to model.compile

  Returns:
    A new model with shared layers with the old model.
  """
    from tensorflow.python.keras import models
    from tensorflow.python.keras.engine import sequential
    if isinstance(model, sequential.Sequential):
        updated_model = models._clone_sequential_model(model, input_tensors=inputs, layer_fn=models.share_weights)
    else:
        updated_model = models._clone_functional_model(model, input_tensors=inputs, layer_fn=models.share_weights)
        updated_model._callable_losses = model._callable_losses

    def _upcast_low_precision_outputs(output):
        if output.dtype == dtypes.bfloat16:
            return math_ops.cast(output, dtypes.float32)
        else:
            return output
    updated_model.outputs = [_upcast_low_precision_outputs(o) for o in updated_model.outputs]
    if isinstance(targets, tuple):
        targets = nest.flatten(targets)
    if mode == ModeKeys.PREDICT and inputs is not None:
        _custom_compile_for_predict(updated_model)
    else:
        updated_model.compile(model.optimizer, model.loss, metrics=metrics_module.clone_metrics(model._compile_metrics), loss_weights=model.loss_weights, sample_weight_mode=model.sample_weight_mode, weighted_metrics=metrics_module.clone_metrics(model._compile_weighted_metrics), target_tensors=targets)
    return updated_model