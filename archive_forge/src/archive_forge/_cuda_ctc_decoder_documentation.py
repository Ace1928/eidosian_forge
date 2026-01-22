from __future__ import annotations
import math
from typing import List, NamedTuple, Union
import torch
import torchaudio
import torchaudio.lib.pybind11_prefixctc as cuctc

        Args:
            log_prob (torch.FloatTensor): GPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; log_softmax(output of acoustic model).
            lengths (dtype torch.int32): GPU tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch.

        Returns:
            List[List[CUCTCHypothesis]]:
                List of sorted best hypotheses for each audio sequence in the batch.
        