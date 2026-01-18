from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
Clone a `Model` and build/compile it with the same settings used before.

  This function can be run in the same graph or in a separate graph from the
  model. When using a separate graph, `in_place_reset` must be `False`.

  Note that, currently, the clone produced from this function may not work with
  TPU DistributionStrategy. Try at your own risk.

  Args:
    model: `tf.keras.Model` object. Can be Functional, Sequential, or
      sub-classed.
    input_tensors: Optional list or dictionary of input tensors to build the
      model upon. If not provided, placeholders will be created.
    target_tensors: Optional list of target tensors for compiling the model. If
      not provided, placeholders will be created.
    custom_objects: Optional dictionary mapping string names to custom classes
      or functions.
    compile_clone: Boolean, whether to compile model clone (default `True`).
    in_place_reset: Boolean, whether to reset the model in place. Only used if
      the model is a subclassed model. In the case of a subclassed model,
      this argument must be set to `True` (default `False`). To restore the
      original model, use the function
      `in_place_subclassed_model_state_restoration(model)`.
    optimizer_iterations: An iterations variable that will be incremented by the
      optimizer if the clone is compiled. This argument is used when a Keras
      model is cloned into an Estimator model function, because Estimators
      create their own global step variable.
    optimizer_config: Optimizer config dictionary or list of dictionary
      returned from `get_config()`. This argument should be defined if
      `clone_and_build_model` is called in a different graph or session from
      the original model, and the optimizer is an instance of `OptimizerV2`.

  Returns:
    Clone of the model.

  Raises:
    ValueError: Cloning fails in the following cases
      - cloning a subclassed model with `in_place_reset` set to False.
      - compiling the clone when the original model has not been compiled.
  