import logging
import os
import tarfile
import zipfile
from typing import Any, List, Optional
import torchaudio
def _load_waveform(root: str, filename: str, exp_sample_rate: int):
    path = os.path.join(root, filename)
    waveform, sample_rate = torchaudio.load(path)
    if exp_sample_rate != sample_rate:
        raise ValueError(f'sample rate should be {exp_sample_rate}, but got {sample_rate}')
    return waveform