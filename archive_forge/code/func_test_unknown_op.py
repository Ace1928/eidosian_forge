import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_unknown_op(self):
    """Test that model inference handles unknown ops.
        This special treatment is to support custom ops.
        See comments in shape inference code for details.
        """
    model = '\n            <ir_version: 7, opset_import: [ "" : 17]>\n            agraph (float[N] x) => (y)\n            {\n                y = SomeUnknownOp (x)\n            }\n        '
    self._check(model)