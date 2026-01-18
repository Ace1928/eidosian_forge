from pathlib import Path
from typing import Tuple, Union
import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
Return the n-th sample in the dataset.

        Args:
            n (int): Index of sample to be loaded.

        Returns:
            (torch.Tensor, int, str):
                torch.Tensor
                    Waveform.
                int
                    Sample rate.
                str
                    File name.
        