import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _from_pydict(cls, pyval, typespec, path_so_far):
    """Converts python dictionary `pyval` to a StructuredTensor with rank=0."""
    if typespec is None:
        fields = dict(((k, cls._from_pyval(v, None, path_so_far + (k,))) for k, v in pyval.items()))
    else:
        spec_shape = typespec._shape
        field_specs = typespec._field_specs
        if not (isinstance(typespec, StructuredTensor.Spec) and spec_shape.rank == 0 and (set(pyval) == set(field_specs))):
            raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, pyval, typespec))
        fields = dict(((k, cls._from_pyval(v, field_specs[k], path_so_far + (k,))) for k, v in pyval.items()))
    return StructuredTensor.from_fields(fields=fields, shape=(), validate=False)