import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.reference as orf
The following model is equivalent to the following function.

    .. code-block:: python

        from onnx importonnx.TensorProto
        from onnx.helper import oh.make_tensor

        from onnxscript import script
        from onnxscript.onnx_opset import opset15 as op
        from onnxscript.onnx_types import FLOAT

        @script()
        def loop_range_cond_only(A: FLOAT["N"]) -> FLOAT["N"]:
            T = A
            cond = op.Constant(value=make_tensor("true",onnx.TensorProto.BOOL, [1], [1]))
            while cond:
                T = T + A
                cond = op.ReduceSum(T) > -10
            return T

        model = loop_range_cond_only.to_model_proto()
    