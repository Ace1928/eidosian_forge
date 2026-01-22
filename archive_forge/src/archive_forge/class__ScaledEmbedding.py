import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _ScaledEmbedding(torch.nn.Module):
    """Make continuous embeddings and boost learning rate

    Args:
        num_embeddings (int): number of embeddings
        embedding_dim (int): embedding dimensions
        scale (float, optional): amount to scale learning rate (Default: 10.0)
        smooth (bool, optional): choose to apply smoothing (Default: ``False``)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float=10.0, smooth: bool=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            weight = weight / torch.arange(1, num_embeddings + 1).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for embedding with scale.
        Args:
            x (torch.Tensor): input tensor of shape `(num_embeddings)`

        Returns:
            (Tensor):
                Embedding output of shape `(num_embeddings, embedding_dim)`
        """
        out = self.embedding(x) * self.scale
        return out