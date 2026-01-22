from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def _get_filenames(model: str) -> _PretrainedFiles:
    if model not in ['librispeech', 'librispeech-3-gram', 'librispeech-4-gram']:
        raise ValueError(f"{model} not supported. Must be one of ['librispeech-3-gram', 'librispeech-4-gram', 'librispeech']")
    prefix = f'decoder-assets/{model}'
    return _PretrainedFiles(lexicon=f'{prefix}/lexicon.txt', tokens=f'{prefix}/tokens.txt', lm=f'{prefix}/lm.bin' if model != 'librispeech' else None)