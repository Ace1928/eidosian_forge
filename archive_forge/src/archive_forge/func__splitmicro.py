from __future__ import annotations
from difflib import SequenceMatcher
from typing import Iterable, Iterator
from kombu import version_info_t
def _splitmicro(micro: str, releaselevel: str='', serial: str='') -> tuple[int, str, str]:
    for index, char in enumerate(micro):
        if not char.isdigit():
            break
    else:
        return (int(micro or 0), releaselevel, serial)
    return (int(micro[:index]), micro[index:], serial)