import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
class LogitGenerator(Module):
    """Generate the logits of masked and unmasked inputs.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        num_classes (int): The number of classes in the labels.
        final_dim (int): Project final representations and targets to `final_dim`.
        skip_masked (bool): If True, skip computing losses over masked frames.
        skip_nomask (bool): If True, skip computing losses over unmasked frames.
    """

    def __init__(self, encoder_embed_dim: int, num_classes: int, final_dim: int, skip_masked: bool, skip_nomask: bool):
        super().__init__()
        self.label_embeddings = Parameter(torch.FloatTensor(num_classes, final_dim))
        torch.nn.init.uniform_(self.label_embeddings)
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask

    def forward(self, x: Tensor, label: Tensor, mask_m: Tensor, mask_u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): The feature representation of the last transformer layer.
            label (Tensor): The label Tensor of dimension `[batch, frame]`.
            mask_m (Tensor): The masked indices of dimension `[batch, frame]`.
            mask_u (Tensor): The unmasked indices of dimension `[batch, frame]`.

        Returns:
            Tensor: The logits of masked frames. Tensor of dimension `[masked_frame, final_dim]`.
            Tensor: The logits of unmasked frames. Tensor of dimension `[unmasked_frame, final_dim]`.
        """
        proj_x = self.final_proj(x)
        if self.skip_masked:
            logit_m = None
        else:
            proj_x_m = proj_x[mask_m]
            label_m = label[mask_m]
            logit_m = _compute_logits(proj_x_m, label_m, self.label_embeddings)
        if self.skip_nomask:
            logit_u = None
        else:
            proj_x_u = proj_x[mask_u]
            label_u = label[mask_u]
            logit_u = _compute_logits(proj_x_u, label_u, self.label_embeddings)
        return (logit_m, logit_u)