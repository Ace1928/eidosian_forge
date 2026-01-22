from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _get_fr_labels():
    return ('|', 'e', 's', 'n', 'i', 't', 'r', 'a', 'o', 'u', 'l', 'd', 'c', 'p', 'm', 'é', 'v', 'q', 'f', 'g', 'b', 'h', 'x', 'à', 'j', 'è', 'y', 'ê', 'z', 'ô', 'k', 'ç', 'œ', 'û', 'ù', 'î', 'â', 'w', 'ï', 'ë', 'ü', 'æ')