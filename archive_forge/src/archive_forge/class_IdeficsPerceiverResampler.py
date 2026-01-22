from typing import Optional, Tuple
import torch
import torch.nn as nn
from .configuration_idefics import IdeficsConfig
class IdeficsPerceiverResampler(nn.Module):

    def __init__(self, config: IdeficsConfig, embed_dim: int, depth: int, n_heads: int, head_dim: int, n_latents: int) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. :param embed_dim: Dimensionality of embeddings being fed
        to the Perceiver Resampler (also dimensionality of latent embeddings *returned* by the Perceiver Resampler.
        Could be e.g., VIT embed_dim, ResNet pool dim, and so on.

        Args:
            config (`IdeficsConfig`): config object
            embed_dim (`int`): The size of each embedding vector
            depth (`int`): Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
            n_heads (`int`): Number of heads in each Transformer block (for multi-headed self-attention).
            head_dim (`int`): Dimensionality of each head projection in the Transformer block.
            n_latents (`int`):
                Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).

        """
        super().__init__()
        self.embed_dim, self.n_heads, self.head_dim, self.n_latents = (embed_dim, n_heads, head_dim, n_latents)
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.embed_dim), requires_grad=True)
        self.intermediate_dim = self.embed_dim * 4 if not hasattr(config.vision_config, 'embed_dim') else config.vision_config.embed_dim * 4
        self.blocks = nn.ModuleList([nn.ModuleList([IdeficsPerceiverAttention(self.embed_dim, self.n_heads, self.head_dim, self.qk_layer_norms), IdeficsMLP(self.intermediate_dim, config)]) for _ in range(depth)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Resample arbitrary length context & *compress* down to self.n_latents latent embeddings"""
        latents = self.latents.repeat(context.shape[0], 1, 1)
        for attn, ff in self.blocks:
            latents = attn(context, latents) + latents
            latents = ff(latents) + latents
        return self.layer_norm(latents)