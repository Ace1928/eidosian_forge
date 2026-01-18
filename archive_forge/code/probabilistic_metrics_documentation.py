from typing import Optional
from typing import Union
import tensorflow.compat.v2 as tf
from keras.src.dtensor import utils as dtensor_utils
from keras.src.losses import binary_crossentropy
from keras.src.losses import categorical_crossentropy
from keras.src.losses import kullback_leibler_divergence
from keras.src.losses import poisson
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.metrics import base_metric
from tensorflow.python.util.tf_export import keras_export
Computes the crossentropy metric between the labels and predictions.

    Use this crossentropy metric when there are two or more label classes.
    We expect labels to be provided as integers. If you want to provide labels
    using `one-hot` representation, please use `CategoricalCrossentropy` metric.
    There should be `# classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      from_logits: (Optional) Whether output is expected to be a logits tensor.
        By default, we consider that output encodes a probability distribution.
      ignore_class: Optional integer. The ID of a class to be ignored during
        metric computation. This is useful, for example, in segmentation
        problems featuring a "void" class (commonly -1 or 255) in segmentation
        maps. By default (`ignore_class=None`), all classes are considered.
      axis: (Optional) The dimension along which entropy is
        computed. Defaults to `-1`.

    Standalone usage:

    >>> # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
    >>> # logits = log(y_pred)
    >>> # softmax = exp(logits) / sum(exp(logits), axis=-1)
    >>> # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    >>> # xent = -sum(y * log(softmax), 1)
    >>> # log(softmax) = [[-2.9957, -0.0513, -16.1181],
    >>> #                [-2.3026, -0.2231, -2.3026]]
    >>> # y_true * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
    >>> # xent = [0.0513, 2.3026]
    >>> # Reduced xent = (0.0513 + 2.3026) / 2
    >>> m = tf.keras.metrics.SparseCategoricalCrossentropy()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result().numpy()
    1.1769392

    >>> m.reset_state()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=tf.constant([0.3, 0.7]))
    >>> m.result().numpy()
    1.6271976

    Usage with `compile()` API:

    ```python
    model.compile(
      optimizer='sgd',
      loss='sparse_categorical_crossentropy',
      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])
    ```
    