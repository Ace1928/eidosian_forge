import warnings
from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.types import core as core_types
from tensorflow.python.util.tf_export import tf_export
def _delete_tracking(self, name):
    """Removes the tracking of name."""
    self._maybe_initialize_trackable()
    if name in self._unconditional_dependency_names:
        del self._unconditional_dependency_names[name]
        for index, (dep_name, _) in enumerate(self._unconditional_checkpoint_dependencies):
            if dep_name == name:
                del self._unconditional_checkpoint_dependencies[index]
                break