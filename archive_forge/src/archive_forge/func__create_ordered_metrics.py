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
def _create_ordered_metrics(self):
    """Cache the flat order needed when returning metrics, for backwards compat."""
    self._metrics_in_order = []
    for output_metrics, output_weighted_metrics in zip(self._metrics, self._weighted_metrics):
        for m in nest.flatten(output_metrics):
            if m is not None:
                self._metrics_in_order.append(m)
        for wm in nest.flatten(output_weighted_metrics):
            if wm is not None:
                self._metrics_in_order.append(wm)