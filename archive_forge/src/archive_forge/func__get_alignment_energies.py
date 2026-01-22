import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _get_alignment_energies(self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor) -> Tensor:
    """Get the alignment vector.

        Args:
            query (Tensor): Decoder output with shape (n_batch, n_mels * n_frames_per_step).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, attention_hidden_dim).
            attention_weights_cat (Tensor): Cumulative and previous attention weights
                with shape (n_batch, 2, max of ``text_lengths``).

        Returns:
            alignment (Tensor): attention weights, it is a tensor with shape (batch, max of ``text_lengths``).
        """
    processed_query = self.query_layer(query.unsqueeze(1))
    processed_attention_weights = self.location_layer(attention_weights_cat)
    energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
    alignment = energies.squeeze(2)
    return alignment