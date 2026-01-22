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
def _get_loss_object(self, loss):
    """Returns a `Loss` object.

    Converts the user-supplied loss to a `Loss` object. Also allows
    `SUM_OVER_BATCH_SIZE` reduction to be used for this loss.

    Args:
      loss: A string, function, or `Loss` object.

    Returns:
      A `Loss` object.
    """
    if loss is None:
        return None
    loss = losses_mod.get(loss)
    if not isinstance(loss, losses_mod.Loss):
        loss_name = get_custom_object_name(loss)
        if loss_name is None:
            raise ValueError('Loss should be a callable, found: {}'.format(loss))
        loss = losses_mod.LossFunctionWrapper(loss, name=loss_name)
    loss._allow_sum_over_batch_size = True
    return loss