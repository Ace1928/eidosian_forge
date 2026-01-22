from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _build_sampler_output(sample_results: List[Tuple[List[int], List[int]]], sampling_metadata: SamplingMetadata, prompt_logprobs: List[Optional[PromptLogprobs]], sample_logprobs: List[SampleLogprobs]) -> SamplerOutput:
    sampler_output = []
    for seq_group, sample_result, group_prompt_logprobs, group_sample_logprobs in zip(sampling_metadata.seq_groups, sample_results, prompt_logprobs, sample_logprobs):
        seq_ids, _ = seq_group
        next_token_ids, parent_ids = sample_result
        seq_outputs = []
        for parent_id, next_token_id, logprobs in zip(parent_ids, next_token_ids, group_sample_logprobs):
            seq_outputs.append(SequenceOutput(seq_ids[parent_id], next_token_id, logprobs))
        sampler_output.append(SequenceGroupOutput(seq_outputs, group_prompt_logprobs))
    return sampler_output