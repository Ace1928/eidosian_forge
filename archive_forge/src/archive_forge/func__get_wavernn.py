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
def _get_wavernn(self, *, dl_kwargs=None):
    model = WaveRNN(**self._wavernn_params)
    url = f'{_BASE_URL}/{self._wavernn_path}'
    dl_kwargs = {} if dl_kwargs is None else dl_kwargs
    state_dict = load_state_dict_from_url(url, **dl_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    return model