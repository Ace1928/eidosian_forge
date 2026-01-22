from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
class MetricAttributes(SerializedAttributes.with_attributes('MetricAttributes', checkpointable_objects=['variables'], functions=[])):
    """Attributes that are added to Metric objects when saved to SavedModel.

  List of all attributes:
    variables: list of all variables
  """
    pass