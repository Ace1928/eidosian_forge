import copy
import enum
import gc
import inspect
import itertools
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.utils import is_accelerate_available, is_torch_available
from ...onnx import remove_duplicate_weights_from_tied_info
from ...onnx import merge_decoders
from ...utils import (
from ...utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ...utils import TRANSFORMERS_MINIMUM_VERSION as GLOBAL_MIN_TRANSFORMERS_VERSION
from ...utils.doc import add_dynamic_docstring
from ...utils.import_utils import check_if_transformers_greater, is_onnx_available, is_onnxruntime_available
from ..base import ExportConfig
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import ModelPatcher, Seq2SeqModelPatcher
def with_behavior(self, behavior: Union[str, ConfigBehavior], use_past: bool=False, use_past_in_inputs: bool=False) -> 'OnnxSeq2SeqConfigWithPast':
    """
        Creates a copy of the current OnnxConfig but with a different `ConfigBehavior` and `use_past` value.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
            use_past (`bool`, defaults to `False`):
                Whether or not the ONNX config to instantiate is for a model using KV cache.
            use_past_in_inputs (`bool`, defaults to `False`):
                Whether the KV cache is to be passed as an input to the ONNX.

        Returns:
            `OnnxSeq2SeqConfigWithPast`
        """
    if isinstance(behavior, str) and (not isinstance(behavior, ConfigBehavior)):
        behavior = ConfigBehavior(behavior)
    onnx_config = self.__class__(self._config, task=self.task, int_dtype=self.int_dtype, float_dtype=self.float_dtype, use_past=use_past, use_past_in_inputs=use_past_in_inputs, behavior=behavior, preprocessors=self._preprocessors, legacy=self.legacy)
    onnx_config.variant = self.variant
    return onnx_config