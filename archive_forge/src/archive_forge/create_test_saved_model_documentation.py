import tensorflow.compat.v2 as tf
from absl import app
from absl import flags
from keras.src import regularizers
from keras.src.testing_infra import test_utils
A binary that creates a serialized SavedModel from a keras model.

This is used in tests to ensure that model serialization is deterministic across
different processes.
