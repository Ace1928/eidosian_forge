import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
A preprocessing layer which crosses features using the "hashing trick".

    This layer performs crosses of categorical features using the "hashing
    trick". Conceptually, the transformation can be thought of as:
    `hash(concatenate(features)) % num_bins`.

    This layer currently only performs crosses of scalar inputs and batches of
    scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and
    `()`.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        num_bins: Number of hash bins.
        output_mode: Specification for the output of the layer. Values can be
            `"int"`, or `"one_hot"` configuring the layer as follows:
            - `"int"`: Return the integer bin indices directly.
            - `"one_hot"`: Encodes each individual element in the input into an
                array the same size as `num_bins`, containing a 1 at the input's
                bin index. Defaults to `"int"`.
        sparse: Boolean. Only applicable to `"one_hot"` mode. If `True`,
            returns a `SparseTensor` instead of a dense `Tensor`.
            Defaults to `False`.
        **kwargs: Keyword arguments to construct a layer.

    Examples:

    **Crossing two scalar features.**

    >>> layer = tf.keras.layers.HashedCrossing(
    ...     num_bins=5)
    >>> feat1 = tf.constant(['A', 'B', 'A', 'B', 'A'])
    >>> feat2 = tf.constant([101, 101, 101, 102, 102])
    >>> layer((feat1, feat2))
    <tf.Tensor: shape=(5,), dtype=int64, numpy=array([1, 4, 1, 1, 3])>

    **Crossing and one-hotting two scalar features.**

    >>> layer = tf.keras.layers.HashedCrossing(
    ...     num_bins=5, output_mode='one_hot')
    >>> feat1 = tf.constant(['A', 'B', 'A', 'B', 'A'])
    >>> feat2 = tf.constant([101, 101, 101, 102, 102])
    >>> layer((feat1, feat2))
    <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
      array([[0., 1., 0., 0., 0.],
             [0., 0., 0., 0., 1.],
             [0., 1., 0., 0., 0.],
             [0., 1., 0., 0., 0.],
             [0., 0., 0., 1., 0.]], dtype=float32)>
    