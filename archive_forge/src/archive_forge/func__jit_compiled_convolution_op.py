import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
@tf.function(jit_compile=True)
def _jit_compiled_convolution_op(self, inputs, kernel):
    return self.convolution_op(inputs, kernel)