import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.convolutional.base_conv import Conv
Abstract base layer for separable nD convolution.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    If `use_bias` is True and a bias initializer is provided,
    it adds a bias vector to the output.
    It then optionally applies an activation function to produce the final
    output.

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: A tuple or list of integers specifying the spatial
        dimensions of the filters. Can be a single integer to specify the same
        value for all spatial dimensions.
      strides: A tuple or list of integers specifying the strides
        of the convolution. Can be a single integer to specify the same value
        for all spatial dimensions.
        Specifying any `stride` value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `num_filters_in * depth_multiplier`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias.
      depthwise_initializer: An initializer for the depthwise convolution kernel
        (see `keras.initializers`). If None, then the default initializer
        ('glorot_uniform') will be used.
      pointwise_initializer: An initializer for the pointwise convolution kernel
        (see `keras.initializers`). If None, then the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer ('zeros') will be used (see `keras.initializers`).
      depthwise_regularizer: Optional regularizer for the depthwise
        convolution kernel.
      pointwise_regularizer: Optional regularizer for the pointwise
        convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      depthwise_constraint: Optional projection function to be applied to the
        depthwise kernel after being updated by an `Optimizer` (e.g. used for
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      pointwise_constraint: Optional projection function to be applied to the
        pointwise kernel after being updated by an `Optimizer`.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
    