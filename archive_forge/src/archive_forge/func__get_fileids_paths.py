import os
from pathlib import Path
from typing import List, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.librispeech import _get_librispeech_metadata
from torchaudio.datasets.utils import _extract_tar
def _get_fileids_paths(path: Path, folders: List[str], _ext_audio: str) -> List[Tuple[str, str]]:
    """Get the file names and the corresponding file paths without `speaker_id`
    and `chapter_id` directories.
    The format of path is like:
        {root}/{_ARCHIVE_NAME}/1h/[0-5]/[clean, other] or
        {root}/{_ARCHIVE_NAME}/9h/[clean, other]

    Args:
        path (Path): Root path to the dataset.
        folders (List[str]): Folders that contain the desired audio files.
        _ext_audio (str): Extension of audio files.

    Returns:
        List[Tuple[str, str]]:
            List of tuples where the first element is the relative path to the audio file.
            The format of relative path is like:
            1h/[0-5]/[clean, other] or 9h/[clean, other]
            The second element is the file name without audio extension.
    """
    path = Path(path)
    files_paths = []
    for folder in folders:
        paths = [p.relative_to(path) for p in path.glob(f'{folder}/*/*/*/*{_ext_audio}')]
        files_paths += [(str(p.parent.parent.parent), str(p.stem)) for p in paths]
    files_paths.sort(key=lambda x: x[0] + x[1])
    return files_paths