import hashlib
import json
from typing import Dict, Tuple
import coremltools as ct  # type: ignore[import]
from coremltools.converters.mil.input_types import TensorType  # type: ignore[import]
from coremltools.converters.mil.mil import types  # type: ignore[import]
from coremltools.models.neural_network import quantization_utils  # type: ignore[import]
import torch
def CompileSpec(inputs, outputs, backend=CoreMLComputeUnit.CPU, allow_low_precision=True, quantization_mode=CoreMLQuantizationMode.NONE, mlmodel_export_path=None):
    return (inputs, outputs, backend, allow_low_precision, quantization_mode, mlmodel_export_path)