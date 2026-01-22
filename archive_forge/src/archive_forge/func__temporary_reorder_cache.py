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
def _temporary_reorder_cache(self, past_key_values, beam_idx):
    """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
    model_class = self.__class__.__name__.lower()
    if isinstance(past_key_values, (tuple, list)):
        past_key_values = self._reorder_cache(past_key_values, beam_idx)
    elif 'bloom' in model_class or 'gptbigcode' in model_class:
        if not isinstance(past_key_values, DynamicCache):
            raise ValueError(f'Using an unsupported cache format with {model_class}. Currently, it only supports the legacy tuple format or `DynamicCache`')
        past_key_values = self._reorder_cache(past_key_values, beam_idx)
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    else:
        past_key_values.reorder_cache(beam_idx)
    return past_key_values