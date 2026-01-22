import copy
import logging
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
import onnxruntime as ort
from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..utils import check_if_transformers_greater
from ..utils.file_utils import validate_file_exists
from ..utils.normalized_config import NormalizedConfigManager
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .base import ORTDecoderForSeq2Seq, ORTEncoder
from .constants import (
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
from huggingface_hub.utils import EntryNotFoundError
class ORTEncoderForVisionEncoderDecoder(ORTEncoder):
    """
    Encoder model for ONNX Runtime inference for VisionEncoderDecoder models.

    Args:
        session (`ort.InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    @add_start_docstrings_to_model_forward(VISION_ENCODER_INPUTS_DOCSTRING)
    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> BaseModelOutput:
        use_torch = isinstance(pixel_values, torch.Tensor)
        self.parent_model.raise_on_numpy_input_io_binding(use_torch)
        if self.parent_model.device.type == 'cuda' and self.parent_model.use_io_binding:
            known_output_shapes = self.compute_encoder_known_output_shapes(pixel_values)
            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(self.session, pixel_values, known_output_shapes=known_output_shapes, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            last_hidden_state = output_buffers['last_hidden_state'].view(output_shapes['last_hidden_state'])
        else:
            if use_torch:
                onnx_inputs = {'pixel_values': pixel_values.cpu().detach().numpy()}
            else:
                onnx_inputs = {'pixel_values': pixel_values}
            outputs = self.session.run(None, onnx_inputs)
            last_hidden_state = outputs[self.output_names['last_hidden_state']]
            if use_torch:
                last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)
        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def compute_encoder_known_output_shapes(self, pixel_values: torch.FloatTensor) -> Dict[str, List[int]]:
        if self.normalized_config.config.model_type == 'donut-swin':
            encoder_sequence_length = self.normalized_config.config.image_size[0] * self.normalized_config.config.image_size[1] // self.normalized_config.config.hidden_size
        elif self.normalized_config.config.model_type in ['vit', 'deit']:
            return None
        else:
            raise ValueError(f'Unsupported encoder model type {self.normalized_config.config.model_type} for ORTForVisionSeq2Seq with IOBinding.Currently supported models are vit, donut-swin and deit.Please submit a PR to add support for this model type.')
        return {'last_hidden_state': [pixel_values.shape[0], encoder_sequence_length, self.normalized_config.config.hidden_size]}