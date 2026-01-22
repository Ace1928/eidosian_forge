import collections
import errno
import gc
import itertools
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
import numpy as np
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
def _CreateConverter(self, run_params, saved_model_dir, conversion_params):
    """Returns a TrtGraphConverter."""
    if run_params.is_v2:
        converter_v2 = trt_convert.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, use_dynamic_shape=run_params.dynamic_shape, dynamic_shape_profile_strategy=self._profile_strategy, **conversion_params._asdict())
        if self._disable_non_trt_optimizers:
            converter_v2._test_only_disable_non_trt_optimizers = True
        return converter_v2
    converter_v1 = trt_convert.TrtGraphConverter(input_saved_model_dir=saved_model_dir, max_batch_size=self.GetMaxBatchSize(run_params), max_workspace_size_bytes=conversion_params.max_workspace_size_bytes, precision_mode=conversion_params.precision_mode, minimum_segment_size=conversion_params.minimum_segment_size, is_dynamic_op=run_params.dynamic_engine, maximum_cached_engines=conversion_params.maximum_cached_engines, use_calibration=conversion_params.use_calibration)
    if self._disable_non_trt_optimizers:
        converter_v1._test_only_disable_non_trt_optimizers = True
    return converter_v1