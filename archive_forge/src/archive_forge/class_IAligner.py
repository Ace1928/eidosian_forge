from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional import TokenSpan
class IAligner(ABC):

    @abstractmethod
    def __call__(self, emission: Tensor, tokens: List[List[int]]) -> List[List[TokenSpan]]:
        """Generate list of time-stamped token sequences

        Args:
            emission (Tensor): Sequence of token probability distributions in log-domain.
                Shape: `(time, tokens)`.
            tokens (list of integer sequence): Tokenized transcript.
                Output from :py:class:`torchaudio.pipelines.Wav2Vec2FABundle.Tokenizer`.

        Returns:
            (list of TokenSpan sequence): Tokens with time stamps and scores.
        """