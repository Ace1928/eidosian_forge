import os
from pathlib import Path
from typing import List, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
def _parse_filesystem(self, root: str, url: str, folder_in_archive: str, download: bool) -> None:
    root = Path(root)
    archive = os.path.basename(url)
    archive = root / archive
    self._path = root / folder_in_archive
    if download:
        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                checksum = _RELEASE_CONFIGS['release1']['checksum']
                download_url_to_file(url, archive, hash_prefix=checksum)
            _extract_tar(archive)
    if not os.path.isdir(self._path):
        raise RuntimeError('Dataset not found. Please use `download=True` to download it.')
    self._walker = sorted((str(p.stem) for p in Path(self._path).glob('*.wav')))