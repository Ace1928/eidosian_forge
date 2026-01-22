import inspect
import logging
import warnings
import tensorflow as tf
from tensorflow.python.eager import context
import pennylane as qml
from pennylane.measurements import Shots
def _recursive_conj(dy):
    if isinstance(dy, (tf.Variable, tf.Tensor)):
        return tf.math.conj(dy)
    return tuple((_recursive_conj(d) for d in dy))