import tensorflow.compat.v2 as tf
from keras.src.optimizers import optimizer
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving.object_registration import register_keras_serializable
from tensorflow.python.util.tf_export import keras_export
Update step given gradient and the associated model variable.