import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
def _get_librispeech_metadata(fileid: str, root: str, folder: str, ext_audio: str, ext_txt: str, blist: List[str]) -> Tuple[str, int, str, int, int, int]:
    blist = blist or []
    speaker_id, chapter_id, utterance_id = fileid.split('-')
    fileid_audio = f'{speaker_id}-{chapter_id}-{utterance_id}'
    filepath = os.path.join(folder, speaker_id, chapter_id, f'{fileid_audio}{ext_audio}')
    file_text = f'{speaker_id}-{chapter_id}{ext_txt}'
    file_text = os.path.join(root, folder, speaker_id, chapter_id, file_text)
    uttblist = []
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(' ', 1)
            if fileid_audio == fileid_text:
                for word in transcript.split():
                    if word in blist and word not in uttblist:
                        uttblist.append(word)
                break
        else:
            raise FileNotFoundError(f'Translation not found for {fileid_audio}')
    return (filepath, SAMPLE_RATE, transcript, int(speaker_id), int(chapter_id), int(utterance_id), uttblist)