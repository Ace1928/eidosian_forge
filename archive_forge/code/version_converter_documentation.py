import onnx
import onnx.onnx_cpp2py_export.version_converter as C  # noqa: N812
from onnx import ModelProto
Convert opset version of the ModelProto.

    Arguments:
        model: Model.
        target_version: Target opset version.

    Returns:
        Converted model.

    Raises:
        RuntimeError when some necessary conversion is not supported.
    