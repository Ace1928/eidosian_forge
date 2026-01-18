import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils as dtensor_utils
from keras.src.losses import logcosh
from keras.src.losses import mean_absolute_error
from keras.src.losses import mean_absolute_percentage_error
from keras.src.losses import mean_squared_error
from keras.src.losses import mean_squared_logarithmic_error
from keras.src.metrics import base_metric
from keras.src.utils import losses_utils
from keras.src.utils import metrics_utils
from keras.src.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.util.tf_export import keras_export
Accumulates root mean squared error statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`. Defaults to `1`.

        Returns:
          Update op.
        