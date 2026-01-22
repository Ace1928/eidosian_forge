from typing import Optional, Tuple
import torch
import torch.nn as nn
from .configuration_idefics import IdeficsConfig
class IdeficsPerceiverAttention(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, head_dim: int, qk_layer_norms: bool) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()
        self.embed_dim, self.n_heads, self.head_dim = (embed_dim, n_heads, head_dim)
        self.qk_layer_norms = qk_layer_norms
        self.context_layer_norm = nn.LayerNorm(self.embed_dim)
        self.latents_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.qk_layer_norms:
            self.q_layer_norm = nn.LayerNorm(self.head_dim)
            self.k_layer_norm = nn.LayerNorm(self.head_dim)
        self.qk_scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(self.n_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            context (`torch.Tensor`):
                Tensor of shape `[bsz, seq, embed_dim]` representing long-form context to resample.
            latents (`torch.Tensor`):
                Tensor of shape `[bsz, n_latents, embed_dim]` representing fixed length latents to compress to.

        Returns:
            `torch.Tensor`: Tensor of shape `[bsz, n_latents, embed_dim]` representing attention over latents w/ cross
            from context.
        """
        context = self.context_layer_norm(context)
        latents = self.latents_layer_norm(latents)
        batch_size, seq_length, embed_dim = context.shape[:3]
        q = self.q_proj(latents)
        k = self.k_proj(torch.cat([context, latents], dim=-2))
        v = self.v_proj(torch.cat([context, latents], dim=-2))
        q, k, v = [x.reshape(batch_size, x.shape[1], self.n_heads, self.head_dim).transpose(1, 2) for x in (q, k, v)]
        if self.qk_layer_norms:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)
        scores = torch.einsum('... i d, ... j d -> ... i j', q * self.qk_scale, k)
        stabilized_scores = scores - scores.amax(dim=-1, keepdim=True).detach()
        attn = stabilized_scores.softmax(dim=-1)
        resampled = torch.einsum('... i j, ... j d -> ... i d', attn, v)
        return self.output_proj(resampled.transpose(1, 2).flatten(-2))