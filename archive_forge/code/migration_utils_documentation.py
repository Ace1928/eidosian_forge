import sys
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
Wraps TF get_seed to make deterministic random generation easier.

            This makes a variable's initialization (and calls that involve
            random number generation) depend only on how many random number
            generations were used in the scope so far, rather than on how many
            unrelated operations the graph contains.

            Returns:
              Random seed tuple.
            