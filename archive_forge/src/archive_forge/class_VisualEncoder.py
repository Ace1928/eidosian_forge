from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
@dataclass(eq=False)
class VisualEncoder(nn.Module):
    dim: int
    patch_size: int
    image_size: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    pooling: str
    num_reg_tokens: int = 0

    def __post_init__(self):
        super().__init__()
        seq_len = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, self.dim, self.patch_size, self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        if self.num_reg_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, self.num_reg_tokens, self.dim))
        self.blocks = nn.Sequential(*[VisualEncoderBlock(self.dim, self.num_heads) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(self.dim, eps=1e-06)
        self.embedding_projection = nn.Linear(self.dim, self.embedding_dim, bias=False)
        self.return_features = False

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(start_dim=2).transpose(2, 1)
        x = x + self.pos_embed
        special_tokens = [self.cls_token.expand(x.shape[0], -1, -1)]
        if self.num_reg_tokens > 0:
            special_tokens.append(self.reg_token.expand(x.shape[0], -1, -1))
        x = torch.cat(special_tokens + [x], dim=1)
        x = self.blocks(x)
        return self.norm(x)

    def forward_embedding(self, x: Tensor) -> Tensor:
        if self.pooling == 'cls':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        return self.embedding_projection(x)

    def forward(self, x: Tensor, return_features: Optional[bool]=None) -> Tensor:
        features = self.forward_features(x)
        embeddings = self.forward_embedding(features)
        return_features = return_features if return_features is not None else self.return_features
        if return_features:
            return (features, embeddings)
        return embeddings