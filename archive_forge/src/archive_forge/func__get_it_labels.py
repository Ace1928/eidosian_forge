from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _get_it_labels():
    return ('|', 'e', 'i', 'a', 'o', 'n', 't', 'r', 'l', 's', 'c', 'd', 'u', 'p', 'm', 'g', 'v', 'h', 'z', 'f', 'b', 'q', 'à', 'è', 'ù', 'é', 'ò', 'ì', 'k', 'y', 'x', 'w', 'j', 'ó', 'í', 'ï')