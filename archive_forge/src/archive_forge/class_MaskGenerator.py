import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
class MaskGenerator(Module):
    """Generate the masks for masked prediction.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        mask_prob (float): Probability for each token to be chosen as start of the span to be masked.
            This will be multiplied by number of timesteps divided by length of mask span to mask
            approximately this percentage of all elements. However due to overlaps, the actual number
            will be smaller (unless no_overlap is True).
        mask_selection (str): How to choose the mask length.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_other (float): Secondary mask argument (used for more complex distributions).
        mask_length (int): The lengths of the mask.
        no_mask_overlap (bool):  Whether to allow masks to overlap.
        mask_min_space (int):  Minimum space between spans (if no overlap is enabled).
        mask_channel_prob (float): The probability of replacing a feature with 0.
        mask_channel_selection (str): How to choose the mask length for channel masking.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_channel_other (float): Secondary mask argument for channel masking(used for more complex distributions).
        mask_channel_length (int): Minimum space between spans (if no overlap is enabled) for channel masking.
        no_mask_channel_overlap (bool):  Whether to allow channel masks to overlap.
        mask_channel_min_space (int): Minimum space between spans for channel masking(if no overlap is enabled).
    """

    def __init__(self, encoder_embed_dim: int, mask_prob: float, mask_selection: str, mask_other: float, mask_length: int, no_mask_overlap: bool, mask_min_space: int, mask_channel_prob: float, mask_channel_selection: str, mask_channel_other: float, mask_channel_length: int, no_mask_channel_overlap: bool, mask_channel_min_space: int):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.mask_length = mask_length
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.mask_channel_length = mask_channel_length
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.mask_embedding = Parameter(torch.FloatTensor(encoder_embed_dim))
        torch.nn.init.uniform_(self.mask_embedding)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): The encoded representations after feature extraction module.
            padding_mask (Tensor or None): The padding mask of the same dimension as shape,
                which will prevent masking padded elements.

        Returns:
            Tensor: The feature representations after masking.
            Tensor: The generated mask indices.
        """
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = _compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=2, no_overlap=self.no_mask_overlap, min_space=self.mask_min_space)
            mask_indices = mask_indices.to(x.device)
            x[mask_indices] = self.mask_embedding.to(x.dtype)
        else:
            mask_indices = None
        if self.mask_channel_prob > 0:
            mask_channel_indices = _compute_mask_indices((B, C), None, self.mask_channel_prob, self.mask_channel_length, self.mask_channel_selection, self.mask_channel_other, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space)
            mask_channel_indices = mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0
        return (x, mask_indices)