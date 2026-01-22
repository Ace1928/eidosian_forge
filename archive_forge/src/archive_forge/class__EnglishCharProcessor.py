import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.functional import mu_law_decoding
from torchaudio.models import Tacotron2, WaveRNN
from torchaudio.transforms import GriffinLim, InverseMelScale
from . import utils
from .interface import Tacotron2TTSBundle
class _EnglishCharProcessor(Tacotron2TTSBundle.TextProcessor):

    def __init__(self):
        super().__init__()
        self._tokens = utils._get_chars()
        self._mapping = {s: i for i, s in enumerate(self._tokens)}

    @property
    def tokens(self):
        return self._tokens

    def __call__(self, texts: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        indices = [[self._mapping[c] for c in t.lower() if c in self._mapping] for t in texts]
        return utils._to_tensor(indices)