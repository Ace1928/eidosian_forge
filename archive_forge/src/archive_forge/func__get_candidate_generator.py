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
def _get_candidate_generator(self, generation_config: GenerationConfig, input_ids: torch.LongTensor, inputs_tensor: torch.Tensor, assistant_model: 'PreTrainedModel', logits_processor: LogitsProcessorList, model_kwargs: Dict) -> CandidateGenerator:
    """
        Returns the candidate generator to be used in `assisted_generation`
        """
    if generation_config.prompt_lookup_num_tokens is not None:
        candidate_generator = PromptLookupCandidateGenerator(num_output_tokens=generation_config.prompt_lookup_num_tokens)
    else:
        candidate_generator = AssistedCandidateGenerator(input_ids=input_ids, assistant_model=assistant_model, generation_config=generation_config, logits_processor=logits_processor, model_kwargs=model_kwargs, inputs_tensor=inputs_tensor)
    return candidate_generator