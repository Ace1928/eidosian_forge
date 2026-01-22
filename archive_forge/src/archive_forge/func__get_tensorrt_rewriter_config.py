import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _get_tensorrt_rewriter_config(conversion_params, is_dynamic_op=None, max_batch_size=None, is_v2=False, disable_non_trt_optimizers=False, use_implicit_batch=True, profile_strategy=PROFILE_STRATEGY_RANGE):
    """Returns a RewriterConfig proto for TRT transformation.

  Args:
    conversion_params: a TrtConversionParams instance.
    is_dynamic_op: whether to use dynamic engines.
    max_batch_size: maximum batch size for static engines.
    is_v2: whether we're getting a RewriterConfig for TF 2.0.
    disable_non_trt_optimizers: Turn off all default Grappler optimizers.
    use_implicit_batch: Whether to use implicit batch or explicit batch.
    profile_strategy: dynamic shape optimization profile strategy.

  Returns:
    A RewriterConfig proto which sets a TensorRTOptimizer to run Grappler.

  Raises:
    TypeError: if any of the parameters are of unexpected type.
    ValueError: if any of the parameters are of unexpected value.
  """
    _check_conversion_params(conversion_params, is_v2=is_v2)
    if is_v2 and is_dynamic_op is not None and (not is_dynamic_op):
        raise ValueError('is_dynamic_op is either None or True for TF2')
    if not is_v2 and is_dynamic_op is None:
        raise ValueError("is_dynamic_op can't be None for TF1")
    if (is_dynamic_op is None or is_dynamic_op) and max_batch_size is not None:
        raise ValueError('max_batch_size has to be None for TF2 or when is_dynamic_op == True in TF1')
    if is_dynamic_op is not None and (not is_dynamic_op) and (not isinstance(max_batch_size, int)):
        raise ValueError('max_batch_size has to be an integer for is_dynamic_op==False in TF1')
    rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
    rewriter_config_with_trt.remapping = False
    rewriter_config_with_trt.experimental_disable_folding_quantization_emulation = trt_utils.is_linked_tensorrt_version_greater_equal(8, 0, 0) or trt_utils.is_loaded_tensorrt_version_greater_equal(8, 0, 0)
    if not disable_non_trt_optimizers:
        rewriter_config_with_trt.optimizers.extend(['pruning', 'debug_stripper', 'layout', 'dependency', 'constfold', 'common_subgraph_elimination'])
    rewriter_config_with_trt.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    optimizer = rewriter_config_with_trt.custom_optimizers.add()
    if not disable_non_trt_optimizers:
        rewriter_config_with_trt.custom_optimizers.add().name = 'constfold'
    optimizer.name = 'TensorRTOptimizer'
    optimizer.parameter_map['minimum_segment_size'].i = conversion_params.minimum_segment_size
    optimizer.parameter_map['max_workspace_size_bytes'].i = conversion_params.max_workspace_size_bytes
    optimizer.parameter_map['precision_mode'].s = _to_bytes(conversion_params.precision_mode)
    optimizer.parameter_map['maximum_cached_engines'].i = conversion_params.maximum_cached_engines
    optimizer.parameter_map['use_calibration'].b = conversion_params.use_calibration
    optimizer.parameter_map['is_dynamic_op'].b = is_dynamic_op
    optimizer.parameter_map['allow_build_at_runtime'].b = conversion_params.allow_build_at_runtime
    if max_batch_size is not None:
        optimizer.parameter_map['max_batch_size'].i = max_batch_size
    optimizer.parameter_map['use_implicit_batch'].b = use_implicit_batch
    if not use_implicit_batch:
        optimizer.parameter_map['profile_strategy'].s = _to_bytes(profile_strategy.lower())
    if disable_non_trt_optimizers:
        trt_utils.disable_non_trt_optimizers_in_rewriter_config(rewriter_config_with_trt)
    return rewriter_config_with_trt