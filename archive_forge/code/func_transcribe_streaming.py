from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
@torch.jit.export
def transcribe_streaming(self, sources: torch.Tensor, source_lengths: torch.Tensor, state: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
    """Applies transcription network to sources in streaming mode.

        B: batch size;
        T: maximum source sequence segment length in batch;
        D: feature dimension of each source sequence frame.

        Args:
            sources (torch.Tensor): source frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing transcription network internal state generated in preceding invocation
                of ``transcribe_streaming``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing transcription network internal state generated in current invocation
                    of ``transcribe_streaming``.
        """
    return self.transcriber.infer(sources, source_lengths, state)