import collections
import re
from tensorflow.python.util import tf_inspect
def validate_restore_function(trackable, registered_name):
    """Validates whether the trackable can be restored with the saver.

  When using a checkpoint saved with a registered saver, that same saver must
  also be also registered when loading. The name of that saver is saved to the
  checkpoint and set in the `registered_name` arg.

  Args:
    trackable: A `Trackable` object.
    registered_name: String name of the expected registered saver. This argument
      should be set using the name saved in a checkpoint.

  Raises:
    ValueError if the saver could not be found, or if the predicate associated
      with the saver does not pass.
  """
    try:
        _saver_registry.name_lookup(registered_name)
    except LookupError:
        raise ValueError(f"Error when restoring object {trackable} from checkpoint. This object was saved using a registered saver named '{registered_name}', but this saver cannot be found in the current context.")
    if not _saver_registry.get_predicate(registered_name)(trackable):
        raise ValueError(f"Object {trackable} was saved with the registered saver named '{registered_name}'. However, this saver cannot be used to restore the object because the predicate does not pass.")