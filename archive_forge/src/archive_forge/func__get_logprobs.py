from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
from vllm.utils import is_neuron
def _get_logprobs(logprobs: torch.Tensor, sampling_metadata: SamplingMetadata, sample_results: List[Tuple[List[int], List[int]]]) -> Tuple[List[Optional[List[Optional[Dict[int, float]]]]], List[List[Dict[int, float]]]]:
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    largest_num_logprobs = 0
    sample_idx = 0
    for i, (seq_group, sample_result) in enumerate(zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        num_parent_seqs = len(seq_ids)
        if i < sampling_metadata.num_prompts and sampling_params.prompt_logprobs is not None:
            largest_num_logprobs = max(largest_num_logprobs, sampling_params.prompt_logprobs)
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[seq_ids[0]].prompt_token_ids
            batched_logprobs_query_seq_indices.extend((sample_idx + j for j in range(prompt_len - 1)))
            batched_logprobs_query_token_indices.extend((token_id for token_id in prompt_tokens[1:]))
            sample_idx += prompt_len - 1
        batched_logprobs_query_seq_indices.extend([sample_idx + parent_id for parent_id in parent_ids])
        batched_logprobs_query_token_indices.extend(next_token_ids)
        if sampling_params.logprobs is not None:
            largest_num_logprobs = max(largest_num_logprobs, sampling_params.logprobs)
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    batched_logprobs_query_result = logprobs[[batched_logprobs_query_seq_indices, batched_logprobs_query_token_indices]]
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = torch.topk(logprobs, largest_num_logprobs, dim=-1)
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
    else:
        top_logprobs, top_token_ids = (None, None)
    batched_logprobs_query_result = batched_logprobs_query_result.cpu()
    result_prompt_logprobs: List[Optional[PromptLogprobs]] = []
    result_sample_logprobs: List[SampleLogprobs] = []
    sample_idx = 0
    query_result_idx = 0
    for i, (seq_group, sample_result) in enumerate(zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        if i < sampling_metadata.num_prompts and sampling_params.prompt_logprobs is not None:
            num_logprobs = sampling_params.prompt_logprobs
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[seq_ids[0]].prompt_token_ids
            group_prompt_logprobs: PromptLogprobs = [None]
            for token_id in prompt_tokens[1:]:
                prompt_logprobs_dict = {token_id: batched_logprobs_query_result[query_result_idx].item()}
                if num_logprobs > 0:
                    prompt_logprobs_dict.update(zip(top_token_ids[sample_idx, :num_logprobs].tolist(), top_logprobs[sample_idx, :num_logprobs].tolist()))
                group_prompt_logprobs.append(prompt_logprobs_dict)
                sample_idx += 1
                query_result_idx += 1
            result_prompt_logprobs.append(group_prompt_logprobs)
        else:
            result_prompt_logprobs.append(None)
        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        group_sample_logprobs: SampleLogprobs = []
        for next_token_id, parent_id in zip(next_token_ids, parent_ids):
            sample_logprobs_dict = {next_token_id: batched_logprobs_query_result[query_result_idx].item()}
            query_result_idx += 1
            if num_logprobs > 0:
                sample_logprobs_dict.update(zip(top_token_ids[sample_idx + parent_id, :num_logprobs].tolist(), top_logprobs[sample_idx + parent_id, :num_logprobs].tolist()))
            group_sample_logprobs.append(sample_logprobs_dict)
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)
    return (result_prompt_logprobs, result_sample_logprobs)