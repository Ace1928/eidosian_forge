import collections
import itertools
import json
import os
import random
import sys
import threading
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend_config
from keras.src.distribute import distribute_coordinator_utils as dc
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.engine import keras_tensor
from keras.src.utils import control_flow_util
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _assign_value_to_variable(variable, value):
    if isinstance(variable, dtensor.DVariable):
        mesh = variable.layout.mesh
        replicate_layout = dtensor.Layout.replicated(rank=variable.shape.rank, mesh=mesh)
        d_value = dtensor.copy_to_mesh(value, replicate_layout)
        d_value = dtensor.relayout(d_value, variable.layout)
        variable.assign(d_value)
    else:
        variable.assign(value)