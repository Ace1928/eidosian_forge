import tensorflow.compat.v2 as tf
from keras.src.optimizers import adam as adam_experimental
from keras.src.optimizers.legacy import adadelta as adadelta_keras_v2
from keras.src.optimizers.legacy import adagrad as adagrad_keras_v2
from keras.src.optimizers.legacy import adam as adam_keras_v2
from keras.src.optimizers.legacy import adamax as adamax_keras_v2
from keras.src.optimizers.legacy import ftrl as ftrl_keras_v2
from keras.src.optimizers.legacy import (
from keras.src.optimizers.legacy import nadam as nadam_keras_v2
from keras.src.optimizers.legacy import rmsprop as rmsprop_keras_v2
A common set of combination with DistributionStrategies and
    Optimizers.