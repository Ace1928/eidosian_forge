import copy
import warnings
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.module import module
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
@trackable.no_automatic_dependency_tracking
def _build_graph_network_for_inferred_shape(self, input_shape, input_dtype=None):
    if input_shape is None or not self.layers:
        return
    if not tf2.enabled() or not ops.executing_eagerly_outside_functions():
        return
    if not self._has_explicit_input_shape and (not self._use_legacy_deferred_behavior):
        input_shape = tuple(input_shape)
        if self._inferred_input_shape is None:
            new_shape = input_shape
        else:
            new_shape = relax_input_shape(self._inferred_input_shape, input_shape)
        if new_shape is not None and new_shape != self._inferred_input_shape:
            with ops.init_scope():
                inputs = input_layer.Input(batch_shape=new_shape, dtype=input_dtype, name=self.layers[0].name + '_input')
                layer_input = inputs
                created_nodes = set()
                for layer in self.layers:
                    clear_previously_created_nodes(layer, self._created_nodes)
                    try:
                        layer_output = layer(layer_input)
                    except:
                        self._use_legacy_deferred_behavior = True
                        return
                    if len(nest.flatten(layer_output)) != 1:
                        raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
                    track_nodes_created_by_last_call(layer, created_nodes)
                    layer_input = layer_output
                    outputs = layer_output
                self._created_nodes = created_nodes
                try:
                    self._init_graph_network(inputs, outputs)
                    self._graph_initialized = True
                except:
                    self._use_legacy_deferred_behavior = True
            self._inferred_input_shape = new_shape