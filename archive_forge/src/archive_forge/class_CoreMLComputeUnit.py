import hashlib
import json
from typing import Dict, Tuple
import coremltools as ct  # type: ignore[import]
from coremltools.converters.mil.input_types import TensorType  # type: ignore[import]
from coremltools.converters.mil.mil import types  # type: ignore[import]
from coremltools.models.neural_network import quantization_utils  # type: ignore[import]
import torch
class CoreMLComputeUnit:
    CPU = 'cpuOnly'
    CPUAndGPU = 'cpuAndGPU'
    ALL = 'all'