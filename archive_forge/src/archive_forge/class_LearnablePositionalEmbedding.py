from dataclasses import dataclass
import torch
from xformers.components.positional_embedding import (
@register_positional_embedding('learnable', LearnablePositionalEmbeddingConfig)
class LearnablePositionalEmbedding(PositionEmbedding):

    def __init__(self, seq_len: int, dim_model: int, add_class_token: bool=False, *_, **__):
        super().__init__()
        self.pos_emb = torch.nn.Parameter(torch.randn(1, seq_len + int(add_class_token), dim_model) * 0.02)
        self.class_token = torch.nn.Parameter(torch.zeros(dim_model)) if add_class_token else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.class_token is not None:
            clf_token = torch.ones(x.shape[0], 1, self.pos_emb.shape[-1], device=x.device) * self.class_token
            x = torch.cat([clf_token, x], dim=1)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x + self.pos_emb