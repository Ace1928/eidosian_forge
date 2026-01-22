import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
    """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
    self._functions.override(opset, func)