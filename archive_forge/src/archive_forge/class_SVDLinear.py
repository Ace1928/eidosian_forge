import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
class SVDLinear(nn.Module, AdaLoraLayer):

    def __init__(self, base_layer: nn.Module, adapter_name: str, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, fan_in_fan_out: bool=False, init_lora_weights: bool=True, **kwargs) -> None:
        super().__init__()
        AdaLoraLayer.__init__(self, base_layer)
        self.get_base_layer().weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool=False, adapter_names: Optional[List[str]]=None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out) * self.scaling[adapter] / (self.ranknum[adapter] + 1e-05)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_E = self.lora_E[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                ranknum = self.ranknum[active_adapter] + 1e-05
                x = x.to(lora_A.dtype)
                result += dropout(x) @ (lora_A * lora_E).T @ lora_B.T * scaling / ranknum
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return 'adalora.' + rep