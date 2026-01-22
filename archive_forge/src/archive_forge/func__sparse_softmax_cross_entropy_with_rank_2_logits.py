import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
def _sparse_softmax_cross_entropy_with_rank_2_logits(logits, labels, name):
    if config.is_op_determinism_enabled():
        log_probs = log_softmax_v2(logits)
        cost = math_ops.negative(array_ops.gather(log_probs, labels, batch_dims=1))
        nan_tensor = constant_op.constant(float('Nan'), dtype=logits.dtype)
        cost_all_nans = array_ops.broadcast_to(nan_tensor, array_ops.shape(cost))
        class_count = math_ops.cast(array_ops.shape(logits)[-1], labels.dtype)
        cost = array_ops.where(math_ops.logical_or(math_ops.less(labels, 0), math_ops.greater_equal(labels, class_count)), cost_all_nans, cost)
    else:
        cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(logits, labels, name=name)
    return cost