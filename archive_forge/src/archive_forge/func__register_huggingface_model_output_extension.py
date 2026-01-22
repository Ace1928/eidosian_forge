from __future__ import annotations
import contextlib
import functools
import inspect
from typing import (
import torch._dynamo
import torch.export as torch_export
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.utils import _pytree as pytree
def _register_huggingface_model_output_extension(self):
    try:
        from transformers import modeling_outputs
    except ImportError as e:
        return

    @_beartype.beartype
    def model_output_flatten(output: modeling_outputs.ModelOutput) -> Tuple[List[Any], pytree.Context]:
        return (list(output.values()), (type(output), list(output.keys())))

    @_beartype.beartype
    def model_output_unflatten(values: List[Any], context: pytree.Context) -> modeling_outputs.ModelOutput:
        output_type, keys = context
        return output_type(**dict(zip(keys, values)))
    named_model_output_classes = inspect.getmembers(modeling_outputs, lambda x: inspect.isclass(x) and issubclass(x, modeling_outputs.ModelOutput) and (x is not modeling_outputs.ModelOutput))
    for _, class_type in named_model_output_classes:
        self.register_pytree_node(class_type, model_output_flatten, model_output_unflatten)