import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _run_onnx_session_with_ortvaluevector(sess: 'onnxruntime.InferenceSession', input_names: Tuple[str, ...], inputs: Tuple[torch.Tensor, ...], input_devices: Tuple['ORTC.OrtDevice', ...], output_names: Tuple[str, ...], outputs: Tuple[torch.Tensor, ...], output_devices: Tuple['ORTC.OrtDevice', ...], preallocate_output: bool) -> Tuple[torch.Tensor, ...]:
    _nvtx_range_push('contiguous')
    inputs = tuple((a.contiguous() for a in inputs))
    _nvtx_range_pop()
    _nvtx_range_push('push_back_batch')
    ort_inputs = _get_ortvalues_from_torch_tensors(inputs, input_devices)
    if preallocate_output:
        pth_outputs = tuple((_to_real_tensor(t) if isinstance(t, FakeTensor) else t for t in outputs))
        ort_outputs = _get_ortvalues_from_torch_tensors(pth_outputs, output_devices)
    else:
        ort_outputs = ORTC.OrtValueVector()
    _nvtx_range_pop()
    _nvtx_range_push('run_with_ortvaluevector')
    run_options = onnxruntime.RunOptions()
    run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')
    sess.run_with_ortvaluevector(run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices)
    _nvtx_range_pop()
    if preallocate_output:
        return pth_outputs
    else:
        _nvtx_range_push('after run_with_ortvaluevector')
        pth_outputs = onnxruntime.training.ortmodule._utils._ortvalues_to_torch_tensor(ort_outputs)
        _nvtx_range_pop()
        return pth_outputs