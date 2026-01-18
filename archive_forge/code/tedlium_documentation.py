import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
dict[str, tuple[str]]: Phonemes. Mapping from word to tuple of phonemes.
        Note that some words have empty phonemes.
        