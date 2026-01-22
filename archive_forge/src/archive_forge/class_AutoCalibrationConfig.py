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
class AutoCalibrationConfig:

    @staticmethod
    def minmax(dataset: Dataset, moving_average: bool=False, averaging_constant: float=0.01) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            moving_average (`bool`):
                Whether to compute the moving average of the minimum and maximum values.
            averaging_constant (`float`):
                The constant smoothing factor to use when computing the moving average of the minimum and maximum
                values.

        Returns:
            The calibration configuration.
        """
        if moving_average and parse(ort_version) < Version('1.11.0'):
            raise NotImplementedError('MinMax calibration using the moving average method is only implemented for onnxruntime >= 1.11.0')
        if moving_average and (not 0 <= averaging_constant <= 1):
            raise ValueError(f'Invalid averaging constant value ({averaging_constant}) should be within [0, 1]')
        return CalibrationConfig(dataset_name=dataset.info.builder_name, dataset_config_name=dataset.info.config_name, dataset_split=str(dataset.split), dataset_num_samples=dataset.num_rows, method=CalibrationMethod.MinMax, moving_average=moving_average, averaging_constant=averaging_constant)

    @staticmethod
    def entropy(dataset: Dataset, num_bins: int=128, num_quantized_bins: int=128) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            num_bins (`int`):
                The number of bins to use when creating the histogram.
            num_quantized_bins (`int`):
                The number of quantized bins used to find the optimal threshold when computing the activations
                quantization ranges.

        Returns:
            The calibration configuration.
        """
        if parse(ort_version) < Version('1.11.0'):
            raise NotImplementedError('Entropy calibration method is only implemented for onnxruntime >= 1.11.0')
        if num_bins <= 0:
            raise ValueError(f'Invalid value num_bins ({num_bins}) should be >= 1')
        if num_quantized_bins <= 0:
            raise ValueError(f'Invalid value num_quantized_bins ({num_quantized_bins}) should be >= 1')
        return CalibrationConfig(dataset_name=dataset.info.builder_name, dataset_config_name=dataset.info.config_name, dataset_split=str(dataset.split), dataset_num_samples=dataset.num_rows, method=CalibrationMethod.Entropy, num_bins=num_bins, num_quantized_bins=num_quantized_bins)

    @staticmethod
    def percentiles(dataset: Dataset, num_bins: int=2048, percentile: float=99.999) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            num_bins (`int`):
                The number of bins to use when creating the histogram.
            percentile (`float`):
                The percentile to use when computing the activations quantization ranges.

        Returns:
            The calibration configuration.
        """
        if parse(ort_version) < Version('1.11.0'):
            raise NotImplementedError('Percentile calibration method is only implemented for onnxruntime >= 1.11.0')
        if num_bins <= 0:
            raise ValueError(f'Invalid value num_bins ({num_bins}) should be >= 1')
        if not 0 <= percentile <= 100:
            raise ValueError(f'Invalid value percentile ({percentile}) should be within  [0, 100]')
        return CalibrationConfig(dataset_name=dataset.info.builder_name, dataset_config_name=dataset.info.config_name, dataset_split=str(dataset.split), dataset_num_samples=dataset.num_rows, method=CalibrationMethod.Percentile, num_bins=num_bins, percentile=percentile)