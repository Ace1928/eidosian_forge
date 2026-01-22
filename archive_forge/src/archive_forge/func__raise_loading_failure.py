import datetime
import io
import json
import tempfile
import warnings
import zipfile
import ml_dtypes
import numpy as np
from keras.src import backend
from keras.src.backend.common import global_state
from keras.src.layers.layer import Layer
from keras.src.losses.loss import Loss
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.utils import file_utils
from keras.src.utils import naming
from keras.src.version import __version__ as keras_version
def _raise_loading_failure(error_msgs, warn_only=False):
    first_key = list(error_msgs.keys())[0]
    ex_trackable, ex_error = error_msgs[first_key]
    msg = f'A total of {len(error_msgs)} objects could not be loaded. Example error message for object {ex_trackable}:\n\n{ex_error}\n\nList of objects that could not be loaded:\n{[x[0] for x in error_msgs.values()]}'
    if warn_only:
        warnings.warn(msg)
    else:
        raise ValueError(msg)