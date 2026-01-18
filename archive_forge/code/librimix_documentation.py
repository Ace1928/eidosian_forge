import os
from pathlib import Path
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
Load the n-th sample from the dataset.

        Args:
            key (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            int:
                Sample rate
            Tensor:
                Mixture waveform
            List of Tensors:
                List of source waveforms
        