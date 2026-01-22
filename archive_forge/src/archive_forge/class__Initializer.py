import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import nest
class _Initializer(resource.CapturableResource):
    """Represents an initialization operation restored from a SavedModel.

  Without this object re-export of imported 1.x SavedModels would omit the
  original SavedModel's initialization procedure.

  Created when `tf.saved_model.load` loads a TF 1.x-style SavedModel with an
  initialization op. This object holds a function that runs the
  initialization. It does not require any manual user intervention;
  `tf.saved_model.save` will see this object and automatically add it to the
  exported SavedModel, and `tf.saved_model.load` runs the initialization
  function automatically.
  """

    def __init__(self, init_fn, asset_paths):
        super(_Initializer, self).__init__()
        self._asset_paths = asset_paths
        self._init_fn = init_fn

    def _create_resource(self):
        return constant_op.constant(1.0)

    def _initialize(self):
        return self._init_fn(*[path.asset_path for path in self._asset_paths])