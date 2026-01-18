from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
Performs beam search for the given input sequence in streaming mode.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (T, D) or (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape () or (1,).
            beam_width (int): beam size to use during search.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing transcription network internal state generated in preceding
                invocation. (Default: ``None``)
            hypothesis (Hypothesis or None): hypothesis from preceding invocation to seed
                search with. (Default: ``None``)

        Returns:
            (List[Hypothesis], List[List[torch.Tensor]]):
                List[Hypothesis]
                    top-``beam_width`` hypotheses found by beam search.
                List[List[torch.Tensor]]
                    list of lists of tensors representing transcription network
                    internal state generated in current invocation.
        