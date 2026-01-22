from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
@runtime_checkable
class OutputAdaptStep(Protocol):
    """A protocol that defines a step in the output adapting process.

    The output adapting process is a sequence of steps that are applied to the
    PyTorch model outputs to transform them into the outputs format produced by the
    exported ONNX model. Each step takes the PyTorch model outputs as arguments and
    returns the transformed outputs.

    This serves as a base formalized construct for the transformation done to model
    output signature by any individual component in the exporter.
    """

    def apply(self, model_outputs: Any, model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Any:
        ...