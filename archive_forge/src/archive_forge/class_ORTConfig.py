import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from packaging.version import Version, parse
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.registry import IntegerOpsRegistry, QDQRegistry, QLinearOpsRegistry
from onnxruntime.transformers.fusion_options import FusionOptions
from ..configuration_utils import BaseConfig
from ..utils import logging
class ORTConfig(BaseConfig):
    """
    ORTConfig is the configuration class handling all the ONNX Runtime parameters related to the ONNX IR model export,
    optimization and quantization parameters.

    Attributes:
        opset (`Optional[int]`, defaults to `None`):
            ONNX opset version to export the model with.
        use_external_data_format (`bool`, defaults to `False`):
            Allow exporting model >= than 2Gb.
        one_external_file (`bool`, defaults to `True`):
            When `use_external_data_format=True`, whether to save all tensors to one external file.
            If false, save each tensor to a file named with the tensor name.
            (Can not be set to `False` for the quantization)
        optimization (`Optional[OptimizationConfig]`, defaults to `None`):
            Specify a configuration to optimize ONNX Runtime model
        quantization (`Optional[QuantizationConfig]`, defaults to `None`):
            Specify a configuration to quantize ONNX Runtime model
    """
    CONFIG_NAME = 'ort_config.json'
    FULL_CONFIGURATION_FILE = 'ort_config.json'

    def __init__(self, opset: Optional[int]=None, use_external_data_format: bool=False, one_external_file: bool=True, optimization: Optional[OptimizationConfig]=None, quantization: Optional[QuantizationConfig]=None, **kwargs):
        super().__init__()
        self.opset = opset
        self.use_external_data_format = use_external_data_format
        self.one_external_file = one_external_file
        self.optimization = self.dataclass_to_dict(optimization)
        self.quantization = self.dataclass_to_dict(quantization)
        self.optimum_version = kwargs.pop('optimum_version', None)

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        if isinstance(config, dict):
            return config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config