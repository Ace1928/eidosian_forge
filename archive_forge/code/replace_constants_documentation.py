from typing import List, Optional, Union
import numpy as np
from onnx import (
from onnx.helper import (
from onnx.numpy_helper import from_array
Replace initializers or constant node by nodes *ConstantOfShape* to reduce the size.

    This reduce the cost to write a unit test about a specific graph structure.

    Args:
        onx: ModelProto
        threshold: every initializer under this threshold is not
            impacted
        ir_version: initializer must be specified as input for
            `ir_version <= 3`, this must be specified if onx is
            :class:`FunctionProto` or :class:`GraphProto`
        use_range: if uses operator *Range* instead of *ConstantOfShape*
            to avoid constant tensors
        value_constant_of_shape: value to use as a value for all nodes
            *ConstantOfShape*, a high value may produce nan or inf
            predictions

    Returns:
        onx, modified ModelProto

    The function is designed so that the function can be reapplied on a modified model
    and either replace *ConstantOfShape* with *Range* operators, either replace the fill value
    for every *ConstantOfShape*.
    