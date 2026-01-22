import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _get_initial_frame(self, memory: Tensor) -> Tensor:
    """Gets all zeros frames to use as the first decoder input.

        Args:
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        Returns:
            decoder_input (Tensor): all zeros frames with shape
                (n_batch, max of ``text_lengths``, ``n_mels * n_frames_per_step``).
        """
    n_batch = memory.size(0)
    dtype = memory.dtype
    device = memory.device
    decoder_input = torch.zeros(n_batch, self.n_mels * self.n_frames_per_step, dtype=dtype, device=device)
    return decoder_input