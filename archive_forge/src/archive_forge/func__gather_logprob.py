import tensorflow as tf
from ....modeling_tf_utils import keras
from ....tf_utils import shape_list
@staticmethod
def _gather_logprob(logprob, target):
    lp_size = shape_list(logprob)
    r = tf.range(lp_size[0], dtype=target.dtype)
    idx = tf.stack([r, target], 1)
    return tf.gather_nd(logprob, idx)