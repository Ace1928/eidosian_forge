import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
def _get_wavs_paths(data_dir):
    wav_dir = data_dir / 'sentences' / 'wav'
    wav_paths = sorted((str(p) for p in wav_dir.glob('*/*.wav')))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths