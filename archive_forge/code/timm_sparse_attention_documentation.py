import torch
from xformers.components.attention.core import scaled_dot_product_attention

    Almost drop-in replacement for timm attention
    but using the sparsity-aware scaled_dot_product_attention from xformers
    