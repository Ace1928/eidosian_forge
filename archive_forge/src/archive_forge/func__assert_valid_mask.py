import tensorflow as tf
from keras.src.utils import tree
def _assert_valid_mask(mask):
    valid = tf.logical_and(tf.logical_not(_has_fully_masked_sequence(mask)), _is_sequence_right_padded(mask))
    tf.Assert(valid, ["You are passing a RNN mask that does not correspond to right-padded sequences, while using cuDNN, which is not supported. With cuDNN, RNN masks can only be used for right-padding, e.g. `[[True, True, False, False]]` would be a valid mask, but any mask that isn't just contiguous `True`'s on the left and contiguous `False`'s on the right would be invalid. You can pass `use_cudnn=False` to your RNN layer to stop using cuDNN (this may be slower)."])