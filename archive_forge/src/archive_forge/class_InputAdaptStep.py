from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
@runtime_checkable
class InputAdaptStep(Protocol):
    """A protocol that defines a step in the input adapting process.

    The input adapting process is a sequence of steps that are applied to the
    PyTorch model inputs to transform them into the inputs format expected by the
    exported ONNX model. Each step takes the PyTorch model inputs as arguments and
    returns the transformed inputs.

    This serves as a base formalized construct for the transformation done to model
    input signature by any individual component in the exporter.
    """

    def apply(self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        ...