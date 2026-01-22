import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
def _get_paired_flist(root: str, veri_test_path: str):
    f_list = []
    with open(veri_test_path, 'r') as f:
        for line in f:
            label, path1, path2 = line.split()
            f_list.append((label, path1, path2))
    return f_list