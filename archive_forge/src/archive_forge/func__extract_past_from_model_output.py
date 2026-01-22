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
from ..utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig
from .logits_process import (
from .stopping_criteria import (
def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool=False):
    past_key_values = None
    if 'past_key_values' in outputs:
        past_key_values = outputs.past_key_values
    elif 'mems' in outputs:
        past_key_values = outputs.mems
    elif 'past_buckets_states' in outputs:
        past_key_values = outputs.past_buckets_states
    if standardize_cache_format and hasattr(self, '_convert_to_standard_cache'):
        batch_size = outputs.logits.shape[0]
        past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
    return past_key_values