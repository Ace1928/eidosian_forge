from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
class OpRunReduceNumpy(OpRun):
    """Implements the reduce logic.
    It must have a parameter *axes*.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRun.__init__(self, onnx_node, run_params)
        if hasattr(self, 'axes'):
            if isinstance(self.axes, np.ndarray):
                if len(self.axes.shape) == 0 or self.axes.shape[0] == 0:
                    self.axes = None
                else:
                    self.axes = tuple(self.axes)
            elif self.axes in [[], ()]:
                self.axes = None
            elif isinstance(self.axes, list):
                self.axes = tuple(self.axes)

    def is_axes_empty(self, axes):
        return axes is None

    def handle_axes(self, axes):
        if isinstance(axes, tuple):
            if len(axes) == 0:
                return None
            return axes
        if axes is None:
            return None
        if isinstance(axes, (int, tuple)):
            return axes
        if not isinstance(axes, np.ndarray):
            raise TypeError(f'axes must be an array, not {type(axes)}.')
        if len(axes.shape) == 0:
            return int(axes)
        if 0 in axes.shape:
            return None
        return tuple(axes.ravel().tolist())

    def output_shape(self, data, axes, keepdims):
        return np.sum(data, axis=axes, keepdims=keepdims).shape

    def reduce_constant(self, data, const_val, axes, keepdims):
        """Special case reduction where the output value is a constant."""
        output_shape = self.output_shape(data, axes, keepdims)
        return (np.full(output_shape, const_val, dtype=data.dtype),)