import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            Dict[str, str]:
                Dictionary containing the following items from the corresponding TSV file;

                * ``"client_id"``
                * ``"path"``
                * ``"sentence"``
                * ``"up_votes"``
                * ``"down_votes"``
                * ``"age"``
                * ``"gender"``
                * ``"accent"``
        