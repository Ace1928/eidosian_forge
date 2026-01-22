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
def _AddTestsFor(test_class, is_v2):
    """Adds test methods to TfTrtIntegrationTestBase for specific TF version."""
    opts = _GetTestConfigsV2() if is_v2 else _GetTestConfigsV1()
    for precision_mode, convert_online, dynamic_engine, use_calibration, dynamic_shape in opts:
        conversion = 'OnlineConversion' if convert_online else 'OfflineConversion'
        engine_type = 'DynamicEngine' if dynamic_engine else 'StaticEngine'
        calibration_type = 'UseCalibration' if use_calibration else 'NoCalibration'
        dynamic_shape_type = 'DynamicShape' if dynamic_shape else 'ImplicitBatch'
        test_name = '%s_%s_%s_%s_%s_%s' % ('testTfTrtV2' if is_v2 else 'testTfTrt', conversion, engine_type, precision_mode, calibration_type, dynamic_shape_type)
        run_params = RunParams(convert_online=convert_online, precision_mode=precision_mode, dynamic_engine=dynamic_engine, test_name=test_name, use_calibration=use_calibration, is_v2=is_v2, dynamic_shape=dynamic_shape)
        if is_v2:
            setattr(test_class, test_name, test_util.run_v2_only(_GetTest(run_params)))
        else:
            setattr(test_class, test_name, test_util.run_v1_only('', _GetTest(run_params)))