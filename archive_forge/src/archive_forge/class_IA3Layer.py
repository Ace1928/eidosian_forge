import warnings
from typing import Any, List, Optional
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose
class IA3Layer(BaseTunerLayer):
    adapter_layer_names = ('ia3_l',)

    def __init__(self, base_layer: nn.Module, is_feedforward: bool, **kwargs) -> None:
        self.base_layer = base_layer
        self.ia3_l = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []
        self.is_feedforward = is_feedforward
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = (base_layer.in_features, base_layer.out_features)
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = (base_layer.in_channels, base_layer.out_channels)
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = (base_layer.num_embeddings, base_layer.embedding_dim)
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = base_layer.weight.ds_shape if hasattr(base_layer.weight, 'ds_shape') else base_layer.weight.shape
        else:
            raise ValueError(f'Unsupported layer type {type(base_layer)}')
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, init_ia3_weights):
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features))
        else:
            weight = torch.randn((self.out_features, 1))
        self.ia3_l[adapter_name] = nn.Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)