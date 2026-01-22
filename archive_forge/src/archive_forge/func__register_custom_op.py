import sys
import warnings
import torch
from torch.onnx import symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args
def _register_custom_op():
    torch.onnx.register_custom_op_symbolic('torchvision::nms', symbolic_multi_label_nms, _ONNX_OPSET_VERSION_11)
    torch.onnx.register_custom_op_symbolic('torchvision::roi_align', roi_align_opset11, _ONNX_OPSET_VERSION_11)
    torch.onnx.register_custom_op_symbolic('torchvision::roi_align', roi_align_opset16, _ONNX_OPSET_VERSION_16)
    torch.onnx.register_custom_op_symbolic('torchvision::roi_pool', roi_pool, _ONNX_OPSET_VERSION_11)