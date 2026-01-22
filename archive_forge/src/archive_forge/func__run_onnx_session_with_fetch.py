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
def _run_onnx_session_with_fetch(sess: 'onnxruntime.InferenceSession', input_names: Tuple[str, ...], inputs: Tuple[torch.Tensor, ...], input_devices: Tuple['ORTC.OrtDevice', ...], output_names: Tuple[str, ...], outputs: Tuple[torch.Tensor, ...], output_devices: Tuple['ORTC.OrtDevice', ...], preallocate_output: bool) -> Tuple[torch.Tensor, ...]:
    feed = {name: onnxruntime.OrtValue.ortvalue_from_numpy(tensor.cpu().numpy()) for name, tensor in zip(input_names, inputs)}
    ort_outputs = sess.run(output_names, feed)
    pth_outputs = tuple((torch.from_numpy(value).to(tensor.device) for value, tensor in zip(ort_outputs, outputs)))
    return pth_outputs