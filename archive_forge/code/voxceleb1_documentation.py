import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform of speaker 1
            Tensor:
                Waveform of speaker 2
            int:
                Sample rate
            int:
                Label
            str:
                File ID of speaker 1
            str:
                File ID of speaker 2
        