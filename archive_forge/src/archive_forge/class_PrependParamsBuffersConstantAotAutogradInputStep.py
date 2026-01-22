from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
class PrependParamsBuffersConstantAotAutogradInputStep(InputAdaptStep):
    """Prepend model parameters, buffers and constants to the user input.

    :func:`torch.export.export` lifts model parameters, buffers and constants as model input, thus, they
    must be added to the user input before the model is executed.

    Args:
        model: The PyTorch model with embedded parameters and buffers.
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
        ordered_params = tuple((model.state_dict[name] for name in model.graph_signature.parameters))
        ordered_buffers = tuple((model.state_dict[name] for name in model.graph_signature.buffers))
        ordered_constant_tensors = tuple((getattr(model.module(), name) for name in model.graph_signature.lifted_tensor_constants))
        updated_args = (*ordered_params, *ordered_buffers, *ordered_constant_tensors, *model_args)
        if model_kwargs:
            return MergeKwargsIntoArgsInputStep().apply(updated_args, model_kwargs, model=model)
        return (updated_args, {})