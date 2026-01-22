from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class OutputAdapter:
    """A class that adapts the PyTorch model outputs to exported ONNX model outputs format."""

    def __init__(self, steps: Optional[List[OutputAdaptStep]]=None):
        self._steps = steps or []

    @_beartype.beartype
    def append_step(self, step: OutputAdaptStep) -> None:
        """Appends a step to the output format steps.

        Args:
            step: The step to append.
        """
        self._steps.append(step)

    @_beartype.beartype
    def apply(self, model_outputs: Any, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Sequence[Union['torch.Tensor', int, float, bool, str]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Args:
            model_outputs: The PyTorch model outputs.
            model: The PyTorch model.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.
        """
        for step in self._steps:
            model_outputs = step.apply(model_outputs, model=model)
        return model_outputs