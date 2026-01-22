from pathlib import Path
from typing import Dict, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip
def _load_dr_vctk_item(self, filename: str) -> Tuple[Tensor, int, Tensor, int, str, str, str, int]:
    speaker_id, utterance_id = filename.split('.')[0].split('_')
    source, channel_id = self._config[filename]
    file_clean_audio = self._clean_audio_dir / filename
    file_noisy_audio = self._noisy_audio_dir / filename
    waveform_clean, sample_rate_clean = torchaudio.load(file_clean_audio)
    waveform_noisy, sample_rate_noisy = torchaudio.load(file_noisy_audio)
    return (waveform_clean, sample_rate_clean, waveform_noisy, sample_rate_noisy, speaker_id, utterance_id, source, channel_id)