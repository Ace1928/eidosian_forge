from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional import TokenSpan
class ITokenizer(ABC):

    @abstractmethod
    def __call__(self, transcript: List[str]) -> List[List[str]]:
        """Tokenize the given transcript (list of word)

        .. note::

           The toranscript must be normalized.

        Args:
            transcript (list of str): Transcript (list of word).

        Returns:
            (list of int): List of token sequences
        """