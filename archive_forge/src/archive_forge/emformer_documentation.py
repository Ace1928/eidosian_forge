import math
from typing import List, Optional, Tuple
import torch
Forward pass for streaming inference.

        B: batch size;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): utterance frames right-padded with right context frames, with
                shape `(B, segment_length + right_context_length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            states (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation of ``infer``. (Default: ``None``)

        Returns:
            (Tensor, Tensor, List[List[Tensor]]):
                Tensor
                    output frames, with shape `(B, segment_length, D)`.
                Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
                List[List[Tensor]]
                    output states; list of lists of tensors representing internal state
                    generated in current invocation of ``infer``.
        