import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
class SaveSliceInfo:
    """Information on how to save this Variable as a slice.

    Provides internal support for saving variables as slices of a larger
    variable.  This API is not public and is subject to change.

    Available properties:

    * full_name
    * full_shape
    * var_offset
    * var_shape
    """

    def __init__(self, full_name=None, full_shape=None, var_offset=None, var_shape=None, save_slice_info_def=None, import_scope=None):
        """Create a `SaveSliceInfo`.

      Args:
        full_name: Name of the full variable of which this `Variable` is a
          slice.
        full_shape: Shape of the full variable, as a list of int.
        var_offset: Offset of this `Variable` into the full variable, as a list
          of int.
        var_shape: Shape of this `Variable`, as a list of int.
        save_slice_info_def: `SaveSliceInfoDef` protocol buffer. If not `None`,
          recreates the SaveSliceInfo object its contents. `save_slice_info_def`
          and other arguments are mutually exclusive.
        import_scope: Optional `string`. Name scope to add. Only used when
          initializing from protocol buffer.
      """
        if save_slice_info_def:
            assert isinstance(save_slice_info_def, variable_pb2.SaveSliceInfoDef)
            self.full_name = ops.prepend_name_scope(save_slice_info_def.full_name, import_scope=import_scope)
            self.full_shape = list(save_slice_info_def.full_shape)
            self.var_offset = list(save_slice_info_def.var_offset)
            self.var_shape = list(save_slice_info_def.var_shape)
        else:
            self.full_name = full_name
            self.full_shape = full_shape
            self.var_offset = var_offset
            self.var_shape = var_shape

    @property
    def spec(self):
        """Computes the spec string used for saving."""
        full_shape_str = ' '.join(('%d' % d for d in self.full_shape)) + ' '
        sl_spec = ':'.join(('%d,%d' % (o, s) for o, s in zip(self.var_offset, self.var_shape)))
        return full_shape_str + sl_spec

    def to_proto(self, export_scope=None):
        """Returns a SaveSliceInfoDef() proto.

      Args:
        export_scope: Optional `string`. Name scope to remove.

      Returns:
        A `SaveSliceInfoDef` protocol buffer, or None if the `Variable` is not
        in the specified name scope.
      """
        if export_scope is None or self.full_name.startswith(export_scope):
            save_slice_info_def = variable_pb2.SaveSliceInfoDef()
            save_slice_info_def.full_name = ops.strip_name_scope(self.full_name, export_scope)
            for i in self.full_shape:
                save_slice_info_def.full_shape.append(i)
            for i in self.var_offset:
                save_slice_info_def.var_offset.append(i)
            for i in self.var_shape:
                save_slice_info_def.var_shape.append(i)
            return save_slice_info_def
        else:
            return None