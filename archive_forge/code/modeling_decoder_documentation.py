import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
import onnxruntime
from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..onnx.utils import check_model_uses_external_data
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ..utils.save_utils import maybe_save_preprocessors
from .constants import DECODER_MERGED_ONNX_FILE_PATTERN, DECODER_ONNX_FILE_PATTERN, DECODER_WITH_PAST_ONNX_FILE_PATTERN
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .models.bloom import bloom_convert_to_bloom_cache, bloom_convert_to_standard_cache
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_WEIGHTS_NAME

        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        