import copy
import itertools
import json
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers
from keras.src.dtensor import dtensor_api
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import compile_utils
from keras.src.engine import data_adapter
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import steps_per_execution_tuning
from keras.src.engine import training_utils
from keras.src.metrics import base_metric
from keras.src.mixed_precision import loss_scale_optimizer as lso
from keras.src.optimizers import optimizer
from keras.src.optimizers import optimizer_v1
from keras.src.saving import pickle_utils
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import model_serialization
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.mode_keys import ModeKeys
def _validate_and_get_metrics_result(self, logs):
    """Returns model metrics as a dict if the keys match with input logs.

        When the training / evalution is performed with asynchronous steps, such
        as the case with `tf.distribute.ParameterServerStrategy`, the last
        scheduled `train / test_step` may not give the latest metrics because it
        is not guaranteed to be executed the last. This method gets metrics from
        the model directly instead of relying on the return from last step
        function.

        It logs a warning if the metric results could not be overridden when
        used with `tf.distribute.ParameterServerStrategy`.

        When the user has custom train / test step functions, the metrics
        returned may be different from `Model.metrics`. In those instances,
        this function will be no-op and return the logs.

        Args:
          logs: A `dict` of metrics returned by train / test step function.

        Returns:
          A `dict` containing values of the metrics listed in `self.metrics`
          when logs and model metrics keys match. Otherwise it returns input
          `logs`.
        """
    PSS_WARN_MSG = 'Could not get Model metric results.         Using the results of last step function could lead to incorrect         results when used with ParameterServerStrategy'
    try:
        metric_logs = self.get_metrics_result()
    except TypeError:
        if self._cluster_coordinator:
            logging.warning(PSS_WARN_MSG)
    else:
        if isinstance(logs, dict) and set(logs.keys()) == set(metric_logs.keys()):
            logs = tf_utils.sync_to_numpy_or_python_type(metric_logs)
        elif self._cluster_coordinator:
            logging.warning(PSS_WARN_MSG)
    return logs