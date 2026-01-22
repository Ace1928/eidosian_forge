import collections
import functools
import itertools
import threading
import warnings
import numpy as np
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_snake_case  # pylint: disable=unused-import
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_tensor_list  # pylint: disable=unused-import
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _symbolic_add_metric(self, value, aggregation=None, name=None):
    base_layer_utils.check_graph_consistency(value, method='add_metric')
    match = self._get_existing_metric(name)
    if aggregation is None:
        if match:
            result_tensor = value
            metric_obj = match
        elif hasattr(value, '_metric_obj'):
            result_tensor = value
            metric_obj = result_tensor._metric_obj
            self._metrics.append(metric_obj)
        else:
            raise ValueError("We do not support adding an aggregated metric result tensor that is not the output of a `tf.keras.metrics.Metric` metric instance. Without having access to the metric instance we cannot reset the state of a metric after every epoch during training. You can create a `tf.keras.metrics.Metric` instance and pass the result here or pass an un-aggregated result with `aggregation` parameter set as `mean`. For example: `self.add_metric(tf.reduce_sum(inputs), name='mean_activation', aggregation='mean')`")
    elif match:
        result_tensor = match(value)
        metric_obj = match
    else:
        metric_obj, result_tensor = base_layer_utils.create_mean_metric(value, name)
        self._metrics.append(metric_obj)