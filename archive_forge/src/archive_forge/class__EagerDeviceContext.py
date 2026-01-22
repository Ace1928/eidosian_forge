import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
class _EagerDeviceContext(object):
    """Context-manager forcing placement of ops and Tensors on a device."""
    __slots__ = ['_device_name', '_ctx', '_stack']

    def __init__(self, ctx, device_name):
        self._device_name = device_name
        self._ctx = ctx
        self._stack = []

    def __enter__(self):
        ctx = self._ctx
        old_device_name = ctx.device_name
        old_device_spec = ctx.device_spec
        new_device_name = self._device_name
        cache_key = (old_device_name, new_device_name)
        try:
            new_device_name, new_device_spec = _device_parsing_cache[cache_key]
        except TypeError:
            raise ValueError('Expecting a string device name. Got %s(%s)' % (type(new_device_name), new_device_name))
        except KeyError:
            if new_device_name is not None:
                if not isinstance(new_device_name, str):
                    raise ValueError('Expecting a string device name. Got %s(%s)' % (type(new_device_name), new_device_name))
                device_spec = pydev.DeviceSpec.from_string(new_device_name)
                if old_device_name:
                    new_device_spec = copy.copy(old_device_spec)
                else:
                    ctx.ensure_initialized()
                    new_device_spec = pydev.DeviceSpec.from_string(ctx._context_devices[0])
                new_device_spec = new_device_spec.make_merged_spec(device_spec)
            else:
                new_device_spec = pydev.DeviceSpec.from_string('')
            new_device_name = new_device_spec.to_string()
            _device_parsing_cache[cache_key] = (new_device_name, new_device_spec)
        ctx._set_device(new_device_name, new_device_spec)
        self._stack.append((old_device_name, old_device_spec, new_device_spec))

    def __exit__(self, *ex_info):
        ctx = self._ctx
        old_device_name, old_device_spec, new_device_spec = self._stack[-1]
        if ctx.device_spec is not new_device_spec:
            raise RuntimeError('Exiting device scope without proper scope nesting')
        del self._stack[-1]
        ctx._set_device(old_device_name, old_device_spec)