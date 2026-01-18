import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def unique_object_name(name, name_uid_map=None, avoid_names=None, namespace='', zero_based=False, avoid_observed_names=False):
    """Makes a object name (or arbitrary string) unique within a TensorFlow graph.

  Args:
    name: String name to make unique.
    name_uid_map: An optional defaultdict(int) to use when creating unique
      names. If None (default), uses a per-Graph dictionary.
    avoid_names: An optional set or dict with names which should not be used. If
      None (default), don't avoid any names unless `avoid_observed_names` is
      True.
    namespace: Gets a name which is unique within the (graph, namespace). Layers
      which are not Networks use a blank namespace and so get graph-global
      names.
    zero_based: If True, name sequences start with no suffix (e.g. "dense",
      "dense_1"). If False, naming is one-based ("dense_1", "dense_2").
    avoid_observed_names: If True, avoid any names that have been observed by
      `backend.observe_object_name`.

  Returns:
    Unique string name.

  Example:


  unique_object_name('dense')  # dense_1
  unique_object_name('dense')  # dense_2

  """
    if name_uid_map is None:
        name_uid_map = get_default_graph_uid_map()
    if avoid_names is None:
        if avoid_observed_names:
            avoid_names = OBSERVED_NAMES
        else:
            avoid_names = set()
    proposed_name = None
    while proposed_name is None or proposed_name in avoid_names:
        name_key = (namespace, name)
        if zero_based:
            number = name_uid_map[name_key]
            if number:
                proposed_name = name + '_' + str(number)
            else:
                proposed_name = name
            name_uid_map[name_key] += 1
        else:
            name_uid_map[name_key] += 1
            proposed_name = name + '_' + str(name_uid_map[name_key])
    return proposed_name