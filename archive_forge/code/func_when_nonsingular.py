import numpy as np
from tensorboard.plugins.histogram import metadata
from tensorboard.plugins.histogram import summary_v2
def when_nonsingular():
    bucket_width = range_ / tf.cast(bucket_count, tf.float64)
    offsets = data - min_
    bucket_indices = tf.cast(tf.floor(offsets / bucket_width), dtype=tf.int32)
    clamped_indices = tf.minimum(bucket_indices, bucket_count - 1)
    one_hots = tf.one_hot(clamped_indices, depth=bucket_count, dtype=tf.float64)
    bucket_counts = tf.cast(tf.reduce_sum(input_tensor=one_hots, axis=0), dtype=tf.float64)
    edges = tf.linspace(min_, max_, bucket_count + 1)
    left_edges = edges[:-1]
    right_edges = edges[1:]
    return tf.transpose(a=tf.stack([left_edges, right_edges, bucket_counts]))