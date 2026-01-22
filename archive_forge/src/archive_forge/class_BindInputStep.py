from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class BindInputStep(InputAdaptStep):
    """Bind the input arguments to the model signature."""

    def __init__(self, model_signature: inspect.Signature):
        self._model_signature = model_signature

    def apply(self, model_args: Sequence[Any], model_kwargs: Mapping[str, Any], model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Bind the input arguments to the model signature.

        We hope the input kwargs will be mapped to bound.args after binding.
        If not, we will raise an error.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs. args is always empty.

        Raises:
            ValueError: If there are keyword-only arguments left after binding args and
                kwargs to model signature.
        """
        bound = self._model_signature.bind(*model_args, **model_kwargs)
        bound.apply_defaults()
        if bound.kwargs:
            raise ValueError('Keyword-only arguments are not supported.')
        return ((), bound.arguments)