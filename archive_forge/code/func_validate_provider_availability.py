import os
import re
from enum import Enum
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from packaging import version
from transformers.utils import logging
import onnxruntime as ort
from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss
from ..utils.import_utils import _is_package_available
def validate_provider_availability(provider: str):
    """
    Ensure the ONNX Runtime execution provider `provider` is available, and raise an error if it is not.

    Args:
        provider (str): Name of an ONNX Runtime execution provider.
    """
    if version.parse(ort.__version__) < version.parse('1.16.0') and os.name != 'nt' and (provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']):
        path_cuda_lib = os.path.join(ort.__path__[0], 'capi', 'libonnxruntime_providers_cuda.so')
        path_trt_lib = os.path.join(ort.__path__[0], 'capi', 'libonnxruntime_providers_tensorrt.so')
        path_dependecy_loading = os.path.join(ort.__path__[0], 'capi', '_ld_preload.py')
        with open(path_dependecy_loading, 'r') as f:
            file_string = f.read()
            if 'ORT_CUDA' not in file_string or 'ORT_TENSORRT' not in file_string:
                if os.path.isfile(path_cuda_lib) and os.path.isfile(path_trt_lib):
                    raise ImportError(f'`onnxruntime-gpu` is installed, but GPU dependencies are not loaded. It is likely there is a conflicting install between `onnxruntime` and `onnxruntime-gpu`. Please install only `onnxruntime-gpu` in order to use {provider}.')
                elif os.path.isfile(path_cuda_lib) and is_onnxruntime_training_available():
                    if provider == 'TensorrtExecutionProvider':
                        raise ImportError(f"Asked to use {provider}, but `onnxruntime-training` package doesn't support {provider}. Please use `CUDAExecutionProvider` instead.")
                else:
                    raise ImportError(f'Asked to use {provider}, but `onnxruntime-gpu` package was not found. Make sure to install `onnxruntime-gpu` package instead of `onnxruntime`.')
            if provider == 'CUDAExecutionProvider':
                if os.environ.get('ORT_CUDA_UNAVAILABLE', '0') == '1':
                    raise ImportError('`onnxruntime-gpu` package is installed, but CUDA requirements could not be loaded. Make sure to meet the required dependencies: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html')
            if provider == 'TensorrtExecutionProvider':
                if os.environ.get('ORT_TENSORRT_UNAVAILABLE', '0') == '1':
                    raise ImportError('`onnxruntime-gpu` package is installed, but TensorRT requirements could not be loaded. Make sure to meet the required dependencies following https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html and https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider .')
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise ValueError(f'Asked to use {provider} as an ONNX Runtime execution provider, but the available execution providers are {available_providers}.')