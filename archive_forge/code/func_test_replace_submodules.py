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
def test_replace_submodules(dist_init, dummy_model):
    model = dummy_model
    model.supported_lora_modules = ['dense1', 'layer1.dense2']
    model.packed_modules_mapping = {}
    manager = LoRAModelManager(model, 1, 1, 1, LoRAConfig(max_lora_rank=8, max_cpu_loras=8, max_loras=8))
    model = manager.model
    assert isinstance(model.get_submodule('dense1'), ColumnParallelLinearWithLoRA)
    assert isinstance(model.get_submodule('layer1.dense1'), ColumnParallelLinearWithLoRA)
    assert isinstance(model.get_submodule('dense2'), RowParallelLinear)
    assert isinstance(model.get_submodule('layer1.dense2'), RowParallelLinearWithLoRA)