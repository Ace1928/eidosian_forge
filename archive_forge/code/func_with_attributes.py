from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
@staticmethod
def with_attributes(name, checkpointable_objects=None, functions=None, copy_from=None):
    """Creates a subclass with all attributes as specified in the arguments.

    Args:
      name: Name of subclass
      checkpointable_objects: List of checkpointable objects to be serialized
        in the SavedModel.
      functions: List of functions to be serialized in the SavedModel.
      copy_from: List of other SerializedAttributes subclasses. The returned
        class will copy checkpoint objects/functions from each subclass.

    Returns:
      Child class with attributes as defined in the `checkpointable_objects`
      and `functions` lists.
    """
    checkpointable_objects = checkpointable_objects or []
    functions = functions or []
    if copy_from is not None:
        for cls in copy_from:
            checkpointable_objects.extend(cls.all_checkpointable_objects)
            functions.extend(cls.all_functions)
    classdict = {'all_checkpointable_objects': set(checkpointable_objects), 'all_functions': set(functions)}
    return type(name, (SerializedAttributes,), classdict)