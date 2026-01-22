from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class ConvertComplexToRealRepresentationInputStep(InputAdaptStep):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).

    """

    def apply(self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Convert complex tensors to float tensors.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs.
        """
        return (tuple((torch.view_as_real(arg) if isinstance(arg, torch.Tensor) and arg.is_complex() else arg for arg in model_args)), model_kwargs)