from functools import lru_cache
from typing import List, Optional
from .constant import COMMON_SAFE_ASCII_CHARACTERS, UNICODE_SECONDARY_RANGE_KEYWORD
from .utils import (
class ArchaicUpperLowerPlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._buf = False
        self._character_count_since_last_sep = 0
        self._successive_upper_lower_count = 0
        self._successive_upper_lower_count_final = 0
        self._character_count = 0
        self._last_alpha_seen = None
        self._current_ascii_only = True

    def eligible(self, character: str) -> bool:
        return True

    def feed(self, character: str) -> None:
        is_concerned = character.isalpha() and is_case_variable(character)
        chunk_sep = is_concerned is False
        if chunk_sep and self._character_count_since_last_sep > 0:
            if self._character_count_since_last_sep <= 64 and character.isdigit() is False and (self._current_ascii_only is False):
                self._successive_upper_lower_count_final += self._successive_upper_lower_count
            self._successive_upper_lower_count = 0
            self._character_count_since_last_sep = 0
            self._last_alpha_seen = None
            self._buf = False
            self._character_count += 1
            self._current_ascii_only = True
            return
        if self._current_ascii_only is True and is_ascii(character) is False:
            self._current_ascii_only = False
        if self._last_alpha_seen is not None:
            if character.isupper() and self._last_alpha_seen.islower() or (character.islower() and self._last_alpha_seen.isupper()):
                if self._buf is True:
                    self._successive_upper_lower_count += 2
                    self._buf = False
                else:
                    self._buf = True
            else:
                self._buf = False
        self._character_count += 1
        self._character_count_since_last_sep += 1
        self._last_alpha_seen = character

    def reset(self) -> None:
        self._character_count = 0
        self._character_count_since_last_sep = 0
        self._successive_upper_lower_count = 0
        self._successive_upper_lower_count_final = 0
        self._last_alpha_seen = None
        self._buf = False
        self._current_ascii_only = True

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0
        return self._successive_upper_lower_count_final / self._character_count