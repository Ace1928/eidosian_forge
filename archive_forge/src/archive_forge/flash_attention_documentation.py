import torch
from .. import cdiv, jit
from .. import language as tl

Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)

Sequence Parallel implementation inspired by HazyResearch
(see https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
