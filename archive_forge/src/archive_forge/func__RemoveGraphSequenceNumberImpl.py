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
def _RemoveGraphSequenceNumberImpl(self, value, expecting_prefix):
    if isinstance(value, str):
        match = re.search('TRTEngineOp_\\d{3,}_', value)
        has_prefix = match and value.startswith(match.group(0))
        assert not expecting_prefix or has_prefix, f'Expect (not expecting_prefix) or has_prefix but got: - expecting_prefix = {expecting_prefix}\n- has_prefix = {has_prefix}'
        if has_prefix:
            parts = value.split('_', maxsplit=2)
            assert len(parts) == 3, f'Incorrect `parts` of length == 3, but got: len({parts}).'
            return parts[0] + '_' + parts[2]
        return value
    elif isinstance(value, collections.abc.Iterable):
        return set((self._RemoveGraphSequenceNumberImpl(nm, expecting_prefix) for nm in value))
    else:
        raise TypeError("'_RemoveGraphSequenceNumberImpl' can only be used on strings or sequence of strings!")