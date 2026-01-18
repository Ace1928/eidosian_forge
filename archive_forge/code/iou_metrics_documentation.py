from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils as dtensor_utils
from keras.src.metrics import base_metric
from tensorflow.python.util.tf_export import keras_export
Accumulates the confusion matrix statistics.

        Before the confusion matrix is updated, the predicted values are
        thresholded to be:
          0 for values that are smaller than the `threshold`
          1 for values that are larger or equal to the `threshold`

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`. Defaults to `1`.

        Returns:
          Update op.
        