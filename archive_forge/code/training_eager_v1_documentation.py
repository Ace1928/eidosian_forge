import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
Calculates the loss for one input batch.

  Args:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.

  Returns:
      Dict with three items:
        'total_loss': single tensor for overall loss,
        'output_losses': list of tensors for loss corresponding to each of the
          model output. Could be a empty list when model has only one output.
        'metrics': list of tensors for metric specified.
  