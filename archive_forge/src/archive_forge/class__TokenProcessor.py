import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
class _TokenProcessor(ABC):

    @abstractmethod
    def __call__(self, tokens: List[int], **kwargs) -> str:
        """Decodes given list of tokens to text sequence.

        Args:
            tokens (List[int]): list of tokens to decode.

        Returns:
            str:
                Decoded text sequence.
        """