import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine import base_preprocessing_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import tf_utils
def _unzip_dataset(ds):
    """Unzip dataset into a list of single element datasets.

    Args:
      ds: A Dataset object.

    Returns:
      A list of Dataset object, each correspond to one of the `element_spec` of
      the input Dataset object.

    Example:

    >>> ds1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> ds2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    >>> ds_zipped_tuple = tf.data.Dataset.zip((ds1, ds2))
    >>> ds_unzipped_tuple = _unzip_dataset(ds_zipped_tuple)
    >>> ds_zipped_dict = tf.data.Dataset.zip({'ds1': ds1, 'ds2': ds2})
    >>> ds_unzipped_dict = _unzip_dataset(ds_zipped_dict)

    Then the two elements of `ds_unzipped_tuple` and `ds_unzipped_dict` are both
    the same as `ds1` and `ds2`.
    """
    element_count = len(tf.nest.flatten(ds.element_spec))
    ds_unzipped = []
    for i in range(element_count):

        def map_fn(*x, j=i):
            return tf.nest.flatten(x)[j]
        ds_unzipped.append(ds.map(map_fn))
    return ds_unzipped