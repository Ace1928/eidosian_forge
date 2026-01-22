from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _prune_hidden_states(hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    return hidden_states.index_select(0, sampling_metadata.selected_token_indices)