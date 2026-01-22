import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.distribute import model_collection_base
from keras.src.optimizers.legacy import gradient_descent
def _get_data_for_simple_models():
    x_train = tf.constant(np.random.rand(1000, 3), dtype=tf.float32)
    y_train = tf.constant(np.random.rand(1000, 5), dtype=tf.float32)
    x_predict = tf.constant(np.random.rand(1000, 3), dtype=tf.float32)
    return (x_train, y_train, x_predict)