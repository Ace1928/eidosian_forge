import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
def _parse_dictionary(lines: Iterable[str], exclude_punctuations: bool) -> List[str]:
    _alt_re = re.compile('\\([0-9]+\\)')
    cmudict: List[Tuple[str, List[str]]] = []
    for line in lines:
        if not line or line.startswith(';;;'):
            continue
        word, phones = line.strip().split('  ')
        if word in _PUNCTUATIONS:
            if exclude_punctuations:
                continue
            if word.startswith('...'):
                word = '...'
            elif word.startswith('--'):
                word = '--'
            else:
                word = word[0]
        word = re.sub(_alt_re, '', word)
        phones = phones.split(' ')
        cmudict.append((word, phones))
    return cmudict