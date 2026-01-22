import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
class _SummaryContextManager:
    """Context manager to implement SummaryWriter.as_default()."""

    def __init__(self, writer, step=None):
        self._writer = writer
        self._step = step
        self._old_writer = None
        self._old_step = None

    def __enter__(self):
        self._old_writer = _summary_state.writer
        _summary_state.writer = self._writer
        if self._step is not None:
            self._old_step = _summary_state.step
            _summary_state.step = self._step
        return self._writer

    def __exit__(self, *exc):
        _summary_state.writer.flush()
        _summary_state.writer = self._old_writer
        if self._step is not None:
            _summary_state.step = self._old_step
        return False