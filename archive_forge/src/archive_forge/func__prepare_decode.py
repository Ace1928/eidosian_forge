import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import torch
import torch.nn as nn
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
def _prepare_decode(self, seq_group_metadata_list: List[SequenceGroupMetadata]) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int], List[int], Set[LoRARequest]]:
    assert len(seq_group_metadata_list) > 0
    input_tokens: List[List[int]] = []
    input_positions: List[List[int]] = []
    slot_mapping: List[List[int]] = []
    context_lens: List[int] = []
    block_tables: List[List[int]] = []
    lora_index_mapping: List[int] = []
    lora_prompt_mapping: List[int] = []
    lora_requests: Set[LoRARequest] = set()
    for seq_group_metadata in seq_group_metadata_list:
        assert not seq_group_metadata.is_prompt
        seq_ids = list(seq_group_metadata.seq_data.keys())
        lora_id = seq_group_metadata.lora_int_id
        if lora_id > 0:
            lora_requests.add(seq_group_metadata.lora_request)
        for seq_id in seq_ids:
            seq_data = seq_group_metadata.seq_data[seq_id]
            generation_token = seq_data.get_last_token_id()
            input_tokens.append([generation_token])
            seq_len = seq_data.get_len()
            position = seq_len - 1
            input_positions.append([position])
            context_len = seq_len if self.sliding_window is None else min(seq_len, self.sliding_window)
            context_lens.append(context_len)
            block_table = seq_group_metadata.block_tables[seq_id]
            block_number = block_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append([slot])
            lora_index_mapping.append([lora_id])
            lora_prompt_mapping.append(lora_id)
            if self.sliding_window is not None:
                sliding_window_blocks = self.sliding_window // self.block_size
                block_table = block_table[-sliding_window_blocks:]
            block_tables.append(block_table)
    batch_size = len(input_tokens)
    max_context_len = max(context_lens)
    use_captured_graph = not self.model_config.enforce_eager and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1] and (max_context_len <= self.max_context_len_to_capture)
    if use_captured_graph:
        graph_batch_size = _get_graph_batch_size(batch_size)
        assert graph_batch_size >= batch_size
        for _ in range(graph_batch_size - batch_size):
            input_tokens.append([])
            input_positions.append([])
            slot_mapping.append([])
            context_lens.append(1)
            block_tables.append([])
        batch_size = graph_batch_size
    input_tokens = _make_tensor_with_pad(input_tokens, max_len=1, pad=0, dtype=torch.long, device=self.device)
    input_positions = _make_tensor_with_pad(input_positions, max_len=1, pad=0, dtype=torch.long, device=self.device)
    slot_mapping = _make_tensor_with_pad(slot_mapping, max_len=1, pad=_PAD_SLOT_ID, dtype=torch.long, device=self.device)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=self.device)
    if use_captured_graph:
        input_block_tables = self.graph_block_tables[:batch_size]
        for i, block_table in enumerate(block_tables):
            if block_table:
                input_block_tables[i, :len(block_table)] = block_table
        block_tables = torch.tensor(input_block_tables, device=self.device)
    else:
        max_block_table_len = max((len(block_table) for block_table in block_tables))
        block_tables = _make_tensor_with_pad(block_tables, max_len=max_block_table_len, pad=0, dtype=torch.int, device=self.device)
    lora_index_mapping = [_pad_to_max(mapping, 1, pad=0) for mapping in lora_index_mapping]
    input_metadata = InputMetadata(is_prompt=False, slot_mapping=slot_mapping, prompt_lens=None, max_seq_len=None, start_loc=None, max_context_len=max_context_len, context_lens=context_lens, block_tables=block_tables, use_cuda_graph=use_captured_graph, kv_cache_dtype=self.kv_cache_dtype)
    return (input_tokens, input_positions, input_metadata, lora_index_mapping, lora_prompt_mapping, lora_requests)