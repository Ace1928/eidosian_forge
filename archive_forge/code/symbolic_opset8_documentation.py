import functools
import warnings
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import _type_utils, errors, symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
Returns a decorator that calls the decorated (higher-order) function with the given parameters.