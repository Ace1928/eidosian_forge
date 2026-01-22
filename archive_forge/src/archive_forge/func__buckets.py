import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.histogram import metadata
from tensorboard.util import lazy_tensor_creator
from tensorboard.util import tensor_util
def _buckets(data, bucket_count=None):
    """Create a TensorFlow op to group data into histogram buckets.

    Arguments:
      data: A `Tensor` of any shape. Must be castable to `float64`.
      bucket_count: Optional non-negative `int` or scalar `int32` `Tensor`,
        defaults to 30.
    Returns:
      A `Tensor` of shape `[k, 3]` and type `float64`. The `i`th row is
      a triple `[left_edge, right_edge, count]` for a single bucket.
      The value of `k` is either `bucket_count` or `0` (when input data
      is empty).
    """
    if bucket_count is None:
        bucket_count = DEFAULT_BUCKET_COUNT
    with tf.name_scope('buckets'):
        tf.debugging.assert_scalar(bucket_count)
        tf.debugging.assert_type(bucket_count, tf.int32)
        bucket_count = tf.math.maximum(0, bucket_count)
        data = tf.reshape(data, shape=[-1])
        data = tf.cast(data, tf.float64)
        data_size = tf.size(input=data)
        is_empty = tf.logical_or(tf.equal(data_size, 0), tf.less_equal(bucket_count, 0))

        def when_empty():
            """When input data is empty or bucket_count is zero.

            1. If bucket_count is specified as zero, an empty tensor of shape
              (0, 3) will be returned.
            2. If the input data is empty, a tensor of shape (bucket_count, 3)
              of all zero values will be returned.
            """
            return tf.zeros((bucket_count, 3), dtype=tf.float64)

        def when_nonempty():
            min_ = tf.reduce_min(input_tensor=data)
            max_ = tf.reduce_max(input_tensor=data)
            range_ = max_ - min_
            has_single_value = tf.equal(range_, 0)

            def when_multiple_values():
                """When input data contains multiple values."""
                bucket_width = range_ / tf.cast(bucket_count, tf.float64)
                offsets = data - min_
                bucket_indices = tf.cast(tf.floor(offsets / bucket_width), dtype=tf.int32)
                clamped_indices = tf.minimum(bucket_indices, bucket_count - 1)
                one_hots = tf.one_hot(clamped_indices, depth=bucket_count, dtype=tf.float64)
                bucket_counts = tf.cast(tf.reduce_sum(input_tensor=one_hots, axis=0), dtype=tf.float64)
                edges = tf.linspace(min_, max_, bucket_count + 1)
                edges = tf.concat([edges[:-1], [max_]], 0)
                left_edges = edges[:-1]
                right_edges = edges[1:]
                return tf.transpose(a=tf.stack([left_edges, right_edges, bucket_counts]))

            def when_single_value():
                """When input data contains a single unique value."""
                edges = tf.fill([bucket_count], max_)
                zeroes = tf.fill([bucket_count], 0)
                bucket_counts = tf.cast(tf.concat([zeroes[:-1], [data_size]], 0)[:bucket_count], dtype=tf.float64)
                return tf.transpose(a=tf.stack([edges, edges, bucket_counts]))
            return tf.cond(has_single_value, when_single_value, when_multiple_values)
        return tf.cond(is_empty, when_empty, when_nonempty)