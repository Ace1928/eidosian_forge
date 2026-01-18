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
def test_packed_loras(dist_init, dummy_model_gate_up):
    model = dummy_model_gate_up
    model.supported_lora_modules = ['gate_up_proj']
    model.packed_modules_mapping = {'gate_up_proj': ['gate_proj', 'up_proj']}
    model_lora = create_packed_lora(1, model, module_name='gate_up_proj', replaced_module_names=['gate_proj', 'up_proj'])
    model_lora1 = create_packed_lora(2, model, module_name='gate_up_proj', replaced_module_names=['gate_proj', 'up_proj'], empty_replaced_module_name='gate_proj')
    manager = LoRAModelManager(model, 2, 2, 2, LoRAConfig(max_lora_rank=8, max_cpu_loras=2, max_loras=2))
    model = manager.model
    assert isinstance(model.get_submodule('gate_up_proj'), MergedColumnParallelLinearWithLoRA)
    assert manager.add_lora(model_lora)
    assert manager.add_lora(model_lora1)
    packed_lora = model_lora.get_lora('gate_up_proj')
    assert packed_lora and isinstance(packed_lora, PackedLoRALayerWeights)
    assert torch.allclose(packed_lora.lora_a[0], model_lora.get_lora('gate_proj').lora_a)
    assert torch.allclose(packed_lora.lora_b[0], model_lora.get_lora('gate_proj').lora_b)
    assert torch.allclose(packed_lora.lora_a[1], model_lora.get_lora('up_proj').lora_a)
    assert torch.allclose(packed_lora.lora_b[1], model_lora.get_lora('up_proj').lora_b)
    packed_lora1 = model_lora1.get_lora('gate_up_proj')
    assert packed_lora1 and isinstance(packed_lora1, PackedLoRALayerWeights)
    assert packed_lora1.lora_a[0] is None
    assert packed_lora1.lora_b[0] is None
    assert torch.allclose(packed_lora1.lora_a[1], model_lora1.get_lora('up_proj').lora_a)
    assert torch.allclose(packed_lora1.lora_b[1], model_lora1.get_lora('up_proj').lora_b)