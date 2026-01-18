import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.histogram import metadata
from tensorboard.util import lazy_tensor_creator
from tensorboard.util import tensor_util
def when_single_value():
    """When input data contains a single unique value."""
    edges = tf.fill([bucket_count], max_)
    zeroes = tf.fill([bucket_count], 0)
    bucket_counts = tf.cast(tf.concat([zeroes[:-1], [data_size]], 0)[:bucket_count], dtype=tf.float64)
    return tf.transpose(a=tf.stack([edges, edges, bucket_counts]))