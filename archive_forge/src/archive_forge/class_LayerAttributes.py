from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
class LayerAttributes(SerializedAttributes.with_attributes('LayerAttributes', checkpointable_objects=['non_trainable_variables', 'layers', 'metrics', 'layer_regularization_losses', 'layer_metrics'], functions=['call_and_return_conditional_losses', 'activity_regularizer_fn'], copy_from=[CommonEndpoints])):
    """Layer checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from CommonEndpoints
    non_trainable_variables: List of non-trainable variables in the layer and
      its sublayers.
    layers: List of all sublayers.
    metrics: List of all metrics in the layer and its sublayers.
    call_and_return_conditional_losses: Function that takes inputs and returns a
      tuple of (outputs of the call function, list of input-dependent losses).
      The list of losses excludes the activity regularizer function, which is
      separate to allow the deserialized Layer object to define a different
      activity regularizer.
    activity_regularizer_fn: Callable that returns the activity regularizer loss
    layer_regularization_losses: List of losses owned only by this layer.
    layer_metrics: List of metrics owned by this layer.
  """