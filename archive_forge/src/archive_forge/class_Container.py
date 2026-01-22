import copy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
class Container(object):
    """Base Container class."""

    def __init__(self, output_names=None):
        self._output_names = output_names

    def build(self, y_pred):
        if self._output_names is None:
            self._output_names = create_pseudo_output_names(y_pred)

    def _conform_to_outputs(self, outputs, struct):
        """Convenience method to conform `struct` to `outputs` structure.

    Mappings performed:

    (1) Map a dict to a list of outputs, using the output names.
    (2) Fill missing keys in a dict w/ `None`s.
    (3) Map a single item to all outputs.

    Args:
      outputs: Model predictions.
      struct: Arbitrary nested structure (e.g. of labels, sample_weights,
        losses, or metrics).

    Returns:
      Mapping of `struct` to `outputs` structure.
    """
        struct = map_to_output_names(outputs, self._output_names, struct)
        struct = map_missing_dict_keys(outputs, struct)
        if not nest.is_nested(struct) and nest.is_nested(outputs):
            struct = nest.map_structure(lambda _: struct, outputs)
        return struct

    def _maybe_broadcast_to_outputs(self, outputs, objects):
        """Determines if losses / metrics should be applied to all outputs.

    NOTE: This method should only be called for Metrics / Losses, not for
    y_true / sample_weight.

    Args:
      outputs: Model predictions.
      objects: Arbitrary nested structure (e.g. of losses or metrics)

    Returns:
      Arbitrary nested structure of objects, maybe copied to each output.

    Applies a Loss / Metric to all outputs.
    """
        if not self._should_broadcast(objects):
            return objects
        should_copy_objects = len(nest.flatten(outputs)) > 1

        def _broadcast_fn():
            if should_copy_objects:
                return nest.map_structure(self._copy_object, objects)
            return objects
        return nest.map_structure(lambda _: _broadcast_fn(), outputs)

    def _should_broadcast(self, objects):
        raise NotImplementedError

    def _copy_object(self, obj):
        raise NotImplementedError