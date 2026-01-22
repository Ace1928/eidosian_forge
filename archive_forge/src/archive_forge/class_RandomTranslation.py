import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
@keras_export('keras.layers.RandomTranslation', 'keras.layers.experimental.preprocessing.RandomTranslation', v1=[])
class RandomTranslation(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly translates images during training.

    This layer will apply random translations to each image during training,
    filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
      height_factor: a float represented as fraction of value, or a tuple of
          size 2 representing lower and upper bound for shifting vertically. A
          negative value means shifting image up, while a positive value means
          shifting image down. When represented as a single positive float, this
          value is used for both the upper and lower bound. For instance,
          `height_factor=(-0.2, 0.3)` results in an output shifted by a random
          amount in the range `[-20%, +30%]`.  `height_factor=0.2` results in an
          output height shifted by a random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
          2 representing lower and upper bound for shifting horizontally. A
          negative value means shifting image left, while a positive value means
          shifting image right. When represented as a single positive float,
          this value is used for both the upper and lower bound. For instance,
          `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%,
          and shifted right by 30%. `width_factor=0.2` results
          in an output height shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
          to the given mode
          (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
              reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
              filling all values beyond the edge with the same constant value
              k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
              wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
              the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.
    """

    def __init__(self, height_factor, width_factor, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomTranslation').set(True)
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError(f'`height_factor` cannot have upper bound less than lower bound, got {height_factor}')
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(f'`height_factor` argument must have values between [-1, 1]. Received: height_factor={height_factor}')
        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError(f'`width_factor` cannot have upper bound less than lower bound, got {width_factor}')
        if abs(self.width_lower) > 1.0 or abs(self.width_upper) > 1.0:
            raise ValueError(f'`width_factor` must have values between [-1, 1], got {width_factor}')
        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)

        def random_translated_inputs(inputs):
            """Translated inputs with random ops."""
            original_shape = inputs.shape
            unbatched = inputs.shape.rank == 3
            if unbatched:
                inputs = tf.expand_dims(inputs, 0)
            inputs_shape = tf.shape(inputs)
            batch_size = inputs_shape[0]
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            height_translate = self._random_generator.random_uniform(shape=[batch_size, 1], minval=self.height_lower, maxval=self.height_upper, dtype=tf.float32)
            height_translate = height_translate * img_hd
            width_translate = self._random_generator.random_uniform(shape=[batch_size, 1], minval=self.width_lower, maxval=self.width_upper, dtype=tf.float32)
            width_translate = width_translate * img_wd
            translations = tf.cast(tf.concat([width_translate, height_translate], axis=1), dtype=tf.float32)
            output = transform(inputs, get_translation_matrix(translations), interpolation=self.interpolation, fill_mode=self.fill_mode, fill_value=self.fill_value)
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output
        if training:
            return random_translated_inputs(inputs)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'height_factor': self.height_factor, 'width_factor': self.width_factor, 'fill_mode': self.fill_mode, 'fill_value': self.fill_value, 'interpolation': self.interpolation, 'seed': self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))