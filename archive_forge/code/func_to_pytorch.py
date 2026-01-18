import logging
import traceback
from typing import TYPE_CHECKING
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper
from ..utils import is_cupy_available, is_onnxruntime_training_available
@staticmethod
def to_pytorch(ort_value: OrtValue) -> torch.Tensor:
    """
        Converts tensors held by OrtValues in ONNX runtime memory buffer to torch tensor.
        """
    if is_onnxruntime_training_available():
        return IOBindingHelper.to_pytorch_via_dlpack(ort_value)
    else:
        try:
            return IOBindingHelper.to_pytorch_via_cupy(ort_value)
        except Exception:
            logging.error(traceback.format_exc())
            logging.info('Unable to access output memory in CUDA, will offload to CPU')
            return IOBindingHelper.to_pytorch_via_numpy(ort_value)