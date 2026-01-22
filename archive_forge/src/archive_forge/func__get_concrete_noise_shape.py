import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from keras.src.backend.common import standardize_dtype
from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed
def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return tf.shape(inputs)
    concrete_inputs_shape = tf.shape(inputs)
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return concrete_noise_shape