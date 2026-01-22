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
def _concat(data):
    """
        Reverse of `_split` function above.
        """
    if any((data is None for data in data)):
        return None
    if isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)
    elif isinstance(data[0], tuple):
        if isinstance(data[0][0], tuple):
            return tuple((tuple((torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))) for i in range(len(data[0]))))
        else:
            return tuple((torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0]))))
    elif isinstance(data[0], (int, float)):
        return torch.tensor(data)
    else:
        raise ValueError(f'Unexpected attribute type: {type(data[0])}')