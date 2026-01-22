import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
class _EagerTensorBase(tensor_lib.Tensor, internal.NativeObject, core_tf_types.Value):
    """Base class for EagerTensor."""

    def __complex__(self) -> complex:
        return complex(self._numpy())

    def __int__(self) -> int:
        return int(self._numpy())

    def __float__(self) -> float:
        return float(self._numpy())

    def __index__(self):
        return self._numpy().__index__()

    def __bool__(self) -> bool:
        return bool(self._numpy())
    __nonzero__ = __bool__

    def __format__(self, format_spec):
        if self._prefer_custom_summarizer():
            return self._summarize_value().__format__(format_spec)
        elif self.dtype.is_numpy_compatible:
            return self._numpy().__format__(format_spec)
        else:
            return '<unprintable>'.__format__(format_spec)

    def __reduce__(self):
        return (convert_to_tensor, (self._numpy(),))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        del memo
        return self

    def __str__(self) -> str:
        return 'tf.Tensor(%s, shape=%s, dtype=%s)' % (value_text(self, is_repr=False), self.shape, self.dtype.name)

    def __repr__(self) -> str:
        return '<tf.Tensor: shape=%s, dtype=%s, %s>' % (self.shape, self.dtype.name, value_text(self, is_repr=True))

    def __len__(self):
        """Returns the length of the first dimension in the Tensor."""
        if not self.shape.ndims:
            raise TypeError('Scalar tensor has no `len()`')
        try:
            return self._shape_tuple()[0]
        except core._NotOkStatusException as e:
            raise core._status_to_exception(e) from None

    def __array__(self, dtype=None):
        a = self._numpy()
        if not dtype:
            return a
        return np.array(a, dtype=dtype)

    def __hash__(self) -> int:
        raise TypeError('Tensor is unhashable. Instead, use tensor.ref() as the key.')

    def _numpy_internal(self) -> NoReturn:
        raise NotImplementedError()

    def _numpy(self):
        try:
            return self._numpy_internal()
        except core._NotOkStatusException as e:
            raise core._status_to_exception(e) from None

    @property
    def dtype(self):
        return dtypes._INTERN_TABLE[self._datatype_enum()]

    def numpy(self):
        """Copy of the contents of this Tensor into a NumPy array or scalar.

    Unlike NumPy arrays, Tensors are immutable, so this method has to copy
    the contents to ensure safety. Use `memoryview` to get a readonly
    view of the contents without doing a copy:

    >>> t = tf.constant([42])
    >>> np.array(memoryview(t))
    array([42], dtype=int32)

    Note that `memoryview` is only zero-copy for Tensors on CPU. If a Tensor
    is on GPU, it will have to be transferred to CPU first in order for
    `memoryview` to work.

    Returns:
      A NumPy array of the same shape and dtype or a NumPy scalar, if this
      Tensor has rank 0.

    Raises:
      ValueError: If the dtype of this Tensor does not have a compatible
        NumPy dtype.
    """
        maybe_arr = self._numpy()
        return maybe_arr.copy() if isinstance(maybe_arr, np.ndarray) else maybe_arr

    @property
    def backing_device(self):
        """Returns the name of the device holding this tensor's memory.

    `.backing_device` is usually the same as `.device`, which returns
    the device on which the kernel of the operation that produced this tensor
    ran. However, some operations can produce tensors on a different device
    (e.g., an operation that executes on the GPU but produces output tensors
    in host memory).
    """
        raise NotImplementedError()

    def _datatype_enum(self) -> NoReturn:
        raise NotImplementedError()

    def _shape_tuple(self) -> NoReturn:
        """The shape of this Tensor, as a tuple.

    This is more performant than tuple(shape().as_list()) as it avoids
    two list and one object creation. Marked private for now as from an API
    perspective, it would be better to have a single performant way of
    getting a shape rather than exposing shape() and shape_tuple()
    (and heaven forbid, shape_list() etc. as well!). Punting on that for now,
    but ideally one would work things out and remove the need for this method.

    Returns:
      tuple with the shape.
    """
        raise NotImplementedError()

    def _rank(self) -> NoReturn:
        """Integer rank of this Tensor.

    Unlike regular Tensors, the rank is always known for EagerTensors.

    This is more performant than len(self._shape_tuple())

    Returns:
      Integer rank
    """
        raise NotImplementedError()

    def _num_elements(self) -> NoReturn:
        """Number of elements of this Tensor.

    Unlike regular Tensors, the number of elements is always known for
    EagerTensors.

    This is more performant than tensor.shape.num_elements

    Returns:
      Long - num elements in the tensor
    """
        raise NotImplementedError()

    def _copy_to_device(self, device_name) -> NoReturn:
        raise NotImplementedError()

    @staticmethod
    def _override_operator(name, func) -> None:
        setattr(_EagerTensorBase, name, func)

    def _copy_nograd(self, ctx=None, device_name=None):
        """Copies tensor to dest device, but doesn't record the operation."""
        if ctx is None:
            ctx = context.context()
        if device_name is None:
            device_name = ctx.device_name
        try:
            ctx.ensure_initialized()
            new_tensor = self._copy_to_device(device_name)
        except core._NotOkStatusException as e:
            raise core._status_to_exception(e) from None
        return new_tensor

    def _copy(self, ctx=None, device_name=None):
        """Copies tensor to dest device."""
        new_tensor = self._copy_nograd(ctx, device_name)
        if context.executing_eagerly():
            self_device = self.device

            def grad_fun(dresult):
                return [dresult._copy(device_name=self_device) if hasattr(dresult, '_copy') else dresult]
            record.record_operation('_copy', [new_tensor], [self], grad_fun)
        return new_tensor

    @property
    def shape(self):
        if self._tensor_shape is None:
            try:
                self._tensor_shape = tensor_shape.TensorShape(self._shape_tuple())
            except core._NotOkStatusException as e:
                raise core._status_to_exception(e) from None
        return self._tensor_shape

    def get_shape(self) -> tensor_shape.TensorShape:
        """Alias of Tensor.shape."""
        return self.shape

    def _shape_as_list(self) -> List[Tuple[int, ...]]:
        """The shape of the tensor as a list."""
        return list(self._shape_tuple())

    @deprecation.deprecated(None, 'Use tf.identity with explicit device placement instead.')
    def cpu(self):
        """A copy of this Tensor with contents backed by host memory."""
        return self._copy(context.context(), 'CPU:0')

    @deprecation.deprecated(None, 'Use tf.identity instead.')
    def gpu(self, gpu_index=0):
        """A copy of this Tensor with contents backed by memory on the GPU.

    Args:
      gpu_index: Identifies which GPU to place the contents on the returned
        Tensor in.

    Returns:
      A GPU-memory backed Tensor object initialized with the same contents
      as this Tensor.
    """
        return self._copy(context.context(), 'GPU:' + str(gpu_index))

    def set_shape(self, shape) -> None:
        if not self.shape.is_compatible_with(shape):
            raise ValueError(f"Tensor's shape {self.shape} is not compatible with supplied shape {shape}.")

    @property
    def op(self):
        raise AttributeError('Tensor.op is undefined when eager execution is enabled.')

    @property
    def graph(self):
        raise AttributeError('Tensor.graph is undefined when eager execution is enabled.')

    @property
    def name(self):
        raise AttributeError('Tensor.name is undefined when eager execution is enabled.')

    @property
    def value_index(self):
        raise AttributeError('Tensor.value_index is undefined when eager execution is enabled.')

    def consumers(self) -> NoReturn:
        raise NotImplementedError('Tensor.consumers is undefined when eager execution is enabled.')

    def _add_consumer(self, consumer) -> NoReturn:
        raise NotImplementedError('_add_consumer not supported when eager execution is enabled.')

    def _as_node_def_input(self) -> NoReturn:
        raise NotImplementedError('_as_node_def_input not supported when eager execution is enabled.')

    def _as_tf_output(self) -> NoReturn:
        raise NotImplementedError('_as_tf_output not supported when eager execution is enabled.')

    def eval(self, feed_dict=None, session=None) -> NoReturn:
        raise NotImplementedError("eval is not supported when eager execution is enabled, is .numpy() what you're looking for?")

    def __tf_tensor__(self, dtype: Optional[dtypes.DType]=None, name: Optional[str]=None) -> tensor_lib.Tensor:
        if not context.executing_eagerly():
            graph = get_default_graph()
            if not graph.building_function:
                raise RuntimeError(_add_error_prefix('Attempting to capture an EagerTensor without building a function.', name=name))
            return graph.capture(self, name=name)
        return super().__tf_tensor__(dtype, name)

    def _capture_as_const(self, name):
        """Capture the EagerTensor to a graph constant tensor."""
        with control_dependencies(None):
            constant_value = tensor_util.constant_value(self)
            if constant_value is None:
                return None
            const_tensor = _create_graph_constant(constant_value, dtype=self.dtype, shape=self.shape, name=name, verify_shape=False, allow_broadcast=True)
        return const_tensor