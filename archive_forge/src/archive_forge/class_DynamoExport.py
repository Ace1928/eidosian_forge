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
class DynamoExport(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(self, aten_graph: Optional[bool]=None):
        super().__init__()
        self.aten_graph = aten_graph or True

    def generate_fx(self, options: exporter.ResolvedExportOptions, model: Union[torch.nn.Module, Callable], model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        dynamo_flatten_output_step = DynamoFlattenOutputStep()
        wrapped_model = _wrap_model_with_output_adapter(model, dynamo_flatten_output_step)
        self.output_adapter.append_step(dynamo_flatten_output_step)
        fake_mode = options.fake_context.fake_mode if options.fake_context else contextlib.nullcontext()
        fx_mode = 'symbolic' if options.dynamic_shapes else 'fake'
        with fake_mode:
            graph_module, graph_guard = torch._dynamo.export(wrapped_model, tracing_mode=fx_mode)(*model_args, **model_kwargs)
        del graph_guard
        torch._dynamo.reset()
        self.input_adapter.append_step(io_adapter.FlattenInputWithTreeSpecValidationInputStep())
        updated_model_args = self.input_adapter.apply(*model_args, model=model, **model_kwargs)
        return self.pre_export_passes(options, model, graph_module, updated_model_args)

    @_beartype.beartype
    def pre_export_passes(self, options: exporter.ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        return exporter.common_pre_export_passes(options, original_model, fx_module, fx_module_args)