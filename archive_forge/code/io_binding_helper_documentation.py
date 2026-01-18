import logging
import traceback
from typing import TYPE_CHECKING
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper
from ..utils import is_cupy_available, is_onnxruntime_training_available

        Returns an IOBinding object for an inference session. This method is for general purpose, if the inputs and outputs
        are determined, you can prepare data buffers directly to avoid tensor transfers across frameworks.
        