import tensorflow as tf
from ....modeling_tf_utils import keras
from ....tf_utils import shape_list
@staticmethod
def _logit(x, W, b, proj=None):
    y = x
    if proj is not None:
        y = tf.einsum('ibd,ed->ibe', y, proj)
    return tf.einsum('ibd,nd->ibn', y, W) + b