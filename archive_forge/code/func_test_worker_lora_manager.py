import os
from typing import List
import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (LRUCacheWorkerLoRAManager,
from vllm.model_executor.layers.linear import RowParallelLinear
def test_worker_lora_manager(llama_2_7b_model_extra_embeddings, sql_lora_files):
    lora_config = LoRAConfig(max_lora_rank=8, max_cpu_loras=4, max_loras=4)
    worker_lora_manager = WorkerLoRAManager(4, 2, llama_2_7b_model_extra_embeddings.unpadded_vocab_size - lora_config.lora_extra_vocab_size, lora_config, torch.device('cuda'), EMBEDDING_MODULES, EMBEDDING_PADDING_MODULES)
    worker_lora_manager.create_lora_manager(llama_2_7b_model_extra_embeddings)
    mapping = LoRAMapping([], [])
    worker_lora_manager.set_active_loras([LoRARequest('1', 1, sql_lora_files), LoRARequest('2', 2, sql_lora_files)], mapping)
    assert worker_lora_manager.list_loras() == {1, 2}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2
    worker_lora_manager.set_active_loras([LoRARequest('1', 1, sql_lora_files), LoRARequest('3', 3, sql_lora_files), LoRARequest('4', 4, sql_lora_files)], mapping)
    assert worker_lora_manager.list_loras() == {1, 3, 4}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 3
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 4
    worker_lora_manager.set_active_loras([LoRARequest('1', 1, sql_lora_files), LoRARequest('2', 2, sql_lora_files), LoRARequest('5', 5, sql_lora_files)], mapping)
    assert worker_lora_manager.list_loras() == {1, 2, 5}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 2
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 5
    worker_lora_manager.set_active_loras([LoRARequest('1', 1, sql_lora_files), LoRARequest('1', 1, sql_lora_files), LoRARequest('1', 1, sql_lora_files)], mapping)
    assert worker_lora_manager.list_loras() == {1}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 1
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] is None
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] is None
    worker_lora_manager.set_active_loras([LoRARequest('6', 6, sql_lora_files), LoRARequest('7', 7, sql_lora_files), LoRARequest('8', 8, sql_lora_files)], mapping)
    assert worker_lora_manager.list_loras() == {6, 7, 8}
    assert worker_lora_manager._lora_manager.lora_index_to_id[0] == 8
    assert worker_lora_manager._lora_manager.lora_index_to_id[1] == 6
    assert worker_lora_manager._lora_manager.lora_index_to_id[2] == 7
    with pytest.raises(RuntimeError):
        worker_lora_manager.set_active_loras([LoRARequest('10', 10, sql_lora_files), LoRARequest('11', 11, sql_lora_files), LoRARequest('12', 12, sql_lora_files), LoRARequest('13', 13, sql_lora_files), LoRARequest('14', 14, sql_lora_files)], mapping)