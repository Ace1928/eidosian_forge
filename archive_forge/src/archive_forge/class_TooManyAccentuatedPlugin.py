from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class TooManyAccentuatedPlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._character_count = 0
        self._accentuated_count = 0

    def eligible(self, character: str) -> bool:
        return character.isalpha()

    def feed(self, character: str) -> None:
        self._character_count += 1
        if is_accentuated(character):
            self._accentuated_count += 1

    def reset(self) -> None:
        self._character_count = 0
        self._accentuated_count = 0

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        ratio_of_accentuation = self._accentuated_count / self._character_count
        return ratio_of_accentuation if ratio_of_accentuation >= 0.35 else 0.0