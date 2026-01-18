import tensorflow.compat.v2 as tf
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.regularization.dropout import Dropout
from tensorflow.python.util.tf_export import keras_export
Spatial 1D version of Dropout.

    This version performs the same function as Dropout, however, it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
    Call arguments:
      inputs: A 3D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    Input shape:
      3D tensor with shape: `(samples, timesteps, channels)`
    Output shape: Same as input.
    References: - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
    