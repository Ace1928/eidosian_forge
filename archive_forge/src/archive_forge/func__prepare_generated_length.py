import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
from .stopping_criteria import (
def _prepare_generated_length(self, generation_config, has_default_max_length, has_default_min_length, model_input_name, input_ids_length, inputs_tensor):
    """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    elif model_input_name == 'inputs_embeds' and input_ids_length != inputs_tensor.shape[1] and (not self.config.is_encoder_decoder):
        generation_config.max_length -= inputs_tensor.shape[1]
    if generation_config.min_new_tokens is not None:
        if not has_default_min_length:
            logger.warning(f'Both `min_new_tokens` (={generation_config.min_new_tokens}) and `min_length`(={generation_config.min_length}) seem to have been set. `min_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
        generation_config.min_length = generation_config.min_new_tokens + input_ids_length
    elif model_input_name == 'inputs_embeds' and input_ids_length != inputs_tensor.shape[1] and (not self.config.is_encoder_decoder):
        generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)
    return generation_config