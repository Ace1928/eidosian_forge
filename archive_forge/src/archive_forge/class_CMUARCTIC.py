import csv
import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
class CMUARCTIC(Dataset):
    """*CMU ARCTIC* :cite:`Kominek03cmuarctic` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional):
            The URL to download the dataset from or the type of the dataset to download.
            (default: ``"aew"``)
            Allowed type values are ``"aew"``, ``"ahw"``, ``"aup"``, ``"awb"``, ``"axb"``, ``"bdl"``,
            ``"clb"``, ``"eey"``, ``"fem"``, ``"gka"``, ``"jmk"``, ``"ksp"``, ``"ljm"``, ``"lnh"``,
            ``"rms"``, ``"rxr"``, ``"slp"`` or ``"slt"``.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"ARCTIC"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """
    _file_text = 'txt.done.data'
    _folder_text = 'etc'
    _ext_audio = '.wav'
    _folder_audio = 'wav'

    def __init__(self, root: Union[str, Path], url: str=URL, folder_in_archive: str=FOLDER_IN_ARCHIVE, download: bool=False) -> None:
        if url in ['aew', 'ahw', 'aup', 'awb', 'axb', 'bdl', 'clb', 'eey', 'fem', 'gka', 'jmk', 'ksp', 'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt']:
            url = 'cmu_us_' + url + '_arctic'
            ext_archive = '.tar.bz2'
            base_url = 'http://www.festvox.org/cmu_arctic/packed/'
            url = os.path.join(base_url, url + ext_archive)
        root = os.fspath(root)
        basename = os.path.basename(url)
        root = os.path.join(root, folder_in_archive)
        if not os.path.isdir(root):
            os.mkdir(root)
        archive = os.path.join(root, basename)
        basename = basename.split('.')[0]
        self._path = os.path.join(root, basename)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive)
        elif not os.path.exists(self._path):
            raise RuntimeError(f"The path {self._path} doesn't exist. Please check the ``root`` path or set `download=True` to download it")
        self._text = os.path.join(self._path, self._folder_text, self._file_text)
        with open(self._text, 'r') as text:
            walker = csv.reader(text, delimiter='\n')
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Utterance ID
        """
        line = self._walker[n]
        return load_cmuarctic_item(line, self._path, self._folder_audio, self._ext_audio)

    def __len__(self) -> int:
        return len(self._walker)