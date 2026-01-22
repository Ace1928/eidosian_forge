from typing import List, Optional
import torch
from vllm.utils import in_wsl
class LoRALayerWeights:
    """LoRA weights for a layer composed of two low rank matrixes."""

    def __init__(self, module_name: str, rank: int, lora_alpha: int, lora_a: torch.Tensor, lora_b: torch.Tensor, embeddings_tensor: Optional[torch.Tensor]=None, scaling: Optional[float]=None) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.embeddings_tensor = embeddings_tensor
        if scaling is None:
            self.scaling = self.lora_alpha / self.rank
        else:
            self.scaling = scaling

    def optimize(self) -> 'LoRALayerWeights':
        """Optimize the LoRA by merging the scaling into lora_b."""
        if self.scaling == 1:
            return
        self.lora_b *= self.scaling
        self.scaling = 1
        return self

    @property
    def input_dim(self) -> int:
        return self.lora_a.shape[0]

    @property
    def output_dim(self) -> int:
        return self.lora_b.shape[1]

    @property
    def is_packed(self) -> bool:
        return False

    @property
    def extra_vocab_size(self) -> int:
        return self.embeddings_tensor.shape[0] if self.embeddings_tensor is not None else 0

    @classmethod
    def create_dummy_lora_weights(cls, module_name: str, input_dim: int, output_dim: int, rank: int, dtype: torch.dtype, device: torch.device, embeddings_tensor_dim: Optional[int]=None) -> 'LoRALayerWeights':
        pin_memory = str(device) == 'cpu' and (not in_wsl())
        lora_a = torch.zeros([input_dim, rank], dtype=dtype, device=device, pin_memory=pin_memory)
        lora_b = torch.zeros([rank, output_dim], dtype=dtype, device=device, pin_memory=pin_memory)
        embeddings_tensor = torch.rand(10, embeddings_tensor_dim, dtype=dtype, device=device, pin_memory=pin_memory) if embeddings_tensor_dim else None
        return cls(module_name, rank=rank, lora_alpha=1, lora_a=lora_a, lora_b=lora_b, embeddings_tensor=embeddings_tensor)