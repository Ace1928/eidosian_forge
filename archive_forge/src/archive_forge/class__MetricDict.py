import abc
import types
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import utils as dtensor_utils
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import keras_tensor
from keras.src.saving.legacy.saved_model import metric_serialization
from keras.src.utils import generic_utils
from keras.src.utils import losses_utils
from keras.src.utils import metrics_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
class _MetricDict(dict):
    """Wrapper for returned dictionary of metrics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metric_obj = None