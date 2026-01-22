import functools
import threading
import weakref
import tensorflow.compat.v1.logging as logging
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import input_spec
from keras.src.mixed_precision import autocast_variable
from keras.src.saving.legacy import saving_utils
from keras.src.saving.legacy.saved_model import constants
from keras.src.saving.legacy.saved_model import load as keras_load
from keras.src.saving.legacy.saved_model import serialized_attributes
from keras.src.saving.legacy.saved_model import utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_utils
from keras.src.utils import version_utils
from keras.src.utils.generic_utils import LazyLoader
def _filter_shards(variables):
    return [var for var in variables if not hasattr(var, '_sharded_container')]