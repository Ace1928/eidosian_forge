import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
def _load_tedlium_item(self, fileid: str, line: int, path: str) -> Tuple[Tensor, int, str, int, int, int]:
    """Loads a TEDLIUM dataset sample given a file name and corresponding sentence name.

        Args:
            fileid (str): File id to identify both text and audio files corresponding to the sample
            line (int): Line identifier for the sample inside the text file
            path (str): Dataset root path

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, talk_id, speaker_id, identifier)``
        """
    transcript_path = os.path.join(path, 'stm', fileid)
    with open(transcript_path + '.stm') as f:
        transcript = f.readlines()[line]
        talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(' ', 6)
    wave_path = os.path.join(path, 'sph', fileid)
    waveform, sample_rate = self._load_audio(wave_path + self._ext_audio, start_time=start_time, end_time=end_time)
    return (waveform, sample_rate, transcript, talk_id, speaker_id, identifier)