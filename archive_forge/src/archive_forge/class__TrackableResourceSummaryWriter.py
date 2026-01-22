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
class _TrackableResourceSummaryWriter(_ResourceSummaryWriter, resource.TrackableResource, metaclass=_MultiMetaclass):
    """A `_ResourceSummaryWriter` subclass that implements `TrackableResource`."""

    def __init__(self, create_fn, init_op_fn, mesh=None):
        resource.TrackableResource.__init__(self, device='/CPU:0')
        self._create_fn = create_fn
        self._init_op_fn = init_op_fn
        _ResourceSummaryWriter.__init__(self, create_fn=lambda: self.resource_handle, init_op_fn=init_op_fn, mesh=mesh)

    def _create_resource(self):
        return self._create_fn()

    def _initialize(self):
        return self._init_op_fn(self.resource_handle)

    def _destroy_resource(self):
        gen_resource_variable_ops.destroy_resource_op(self.resource_handle, ignore_lookup_error=True)

    def _set_up_resource_deleter(self):
        pass