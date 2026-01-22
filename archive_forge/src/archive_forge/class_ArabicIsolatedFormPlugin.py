from functools import lru_cache
from logging import getLogger
from typing import List, Optional
from .constant import (
from .utils import (
class ArabicIsolatedFormPlugin(MessDetectorPlugin):

    def __init__(self) -> None:
        self._character_count: int = 0
        self._isolated_form_count: int = 0

    def reset(self) -> None:
        self._character_count = 0
        self._isolated_form_count = 0

    def eligible(self, character: str) -> bool:
        return is_arabic(character)

    def feed(self, character: str) -> None:
        self._character_count += 1
        if is_arabic_isolated_form(character):
            self._isolated_form_count += 1

    @property
    def ratio(self) -> float:
        if self._character_count < 8:
            return 0.0
        isolated_form_usage: float = self._isolated_form_count / self._character_count
        return isolated_form_usage