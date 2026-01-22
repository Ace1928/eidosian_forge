import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _do_lstm_arguments_support_cudnn(activation, recurrent_activation, unroll, use_bias):
    from keras.src import activations
    from keras.src import ops
    return activation in (activations.tanh, tf.tanh, ops.tanh) and recurrent_activation in (activations.sigmoid, tf.sigmoid, ops.sigmoid) and (not unroll) and use_bias