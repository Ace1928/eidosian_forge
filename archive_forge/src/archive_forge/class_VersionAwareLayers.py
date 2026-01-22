from tensorflow.python import tf2
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.advanced_activations import PReLU
from tensorflow.python.keras.layers.advanced_activations import ELU
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.advanced_activations import ThresholdedReLU
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import Conv1DTranspose
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.layers.convolutional import Conv3DTranspose
from tensorflow.python.keras.layers.convolutional import SeparableConv1D
from tensorflow.python.keras.layers.convolutional import SeparableConv2D
from tensorflow.python.keras.layers.convolutional import Convolution1D
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.convolutional import Convolution3D
from tensorflow.python.keras.layers.convolutional import Convolution2DTranspose
from tensorflow.python.keras.layers.convolutional import Convolution3DTranspose
from tensorflow.python.keras.layers.convolutional import SeparableConvolution1D
from tensorflow.python.keras.layers.convolutional import SeparableConvolution2D
from tensorflow.python.keras.layers.convolutional import DepthwiseConv2D
from tensorflow.python.keras.layers.convolutional import UpSampling1D
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.convolutional import UpSampling3D
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding3D
from tensorflow.python.keras.layers.convolutional import Cropping1D
from tensorflow.python.keras.layers.convolutional import Cropping2D
from tensorflow.python.keras.layers.convolutional import Cropping3D
from tensorflow.python.keras.layers.core import Masking
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import SpatialDropout1D
from tensorflow.python.keras.layers.core import SpatialDropout2D
from tensorflow.python.keras.layers.core import SpatialDropout3D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.keras.layers.core import Permute
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import ActivityRegularization
from tensorflow.python.keras.layers.dense_attention import AdditiveAttention
from tensorflow.python.keras.layers.dense_attention import Attention
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.merge import Subtract
from tensorflow.python.keras.layers.merge import Multiply
from tensorflow.python.keras.layers.merge import Average
from tensorflow.python.keras.layers.merge import Maximum
from tensorflow.python.keras.layers.merge import Minimum
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.merge import Dot
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.layers.merge import subtract
from tensorflow.python.keras.layers.merge import multiply
from tensorflow.python.keras.layers.merge import average
from tensorflow.python.keras.layers.merge import maximum
from tensorflow.python.keras.layers.merge import minimum
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers.merge import dot
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling3D
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.layers.pooling import MaxPool3D
from tensorflow.python.keras.layers.pooling import AvgPool1D
from tensorflow.python.keras.layers.pooling import AvgPool2D
from tensorflow.python.keras.layers.pooling import AvgPool3D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool1D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool2D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool3D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool3D
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.recurrent import AbstractRNNCell
from tensorflow.python.keras.layers.recurrent import StackedRNNCells
from tensorflow.python.keras.layers.recurrent import SimpleRNNCell
from tensorflow.python.keras.layers.recurrent import PeepholeLSTMCell
from tensorflow.python.keras.layers.recurrent import SimpleRNN
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DeviceWrapper
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DropoutWrapper
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import ResidualWrapper
from tensorflow.python.keras.layers import serialization
from tensorflow.python.keras.layers.serialization import deserialize
from tensorflow.python.keras.layers.serialization import serialize
class VersionAwareLayers(object):
    """Utility to be used internally to access layers in a V1/V2-aware fashion.

  When using layers within the Keras codebase, under the constraint that
  e.g. `layers.BatchNormalization` should be the `BatchNormalization` version
  corresponding to the current runtime (TF1 or TF2), do not simply access
  `layers.BatchNormalization` since it would ignore e.g. an early
  `compat.v2.disable_v2_behavior()` call. Instead, use an instance
  of `VersionAwareLayers` (which you can use just like the `layers` module).
  """

    def __getattr__(self, name):
        serialization.populate_deserializable_objects()
        if name in serialization.LOCAL.ALL_OBJECTS:
            return serialization.LOCAL.ALL_OBJECTS[name]
        return super(VersionAwareLayers, self).__getattr__(name)