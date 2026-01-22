from typing import Tuple, Union
from .big5freq import (
from .euckrfreq import (
from .euctwfreq import (
from .gb2312freq import (
from .jisfreq import (
from .johabfreq import JOHAB_TO_EUCKR_ORDER_TABLE
class CharDistributionAnalysis:
    ENOUGH_DATA_THRESHOLD = 1024
    SURE_YES = 0.99
    SURE_NO = 0.01
    MINIMUM_DATA_THRESHOLD = 3

    def __init__(self) -> None:
        self._char_to_freq_order: Tuple[int, ...] = tuple()
        self._table_size = 0
        self.typical_distribution_ratio = 0.0
        self._done = False
        self._total_chars = 0
        self._freq_chars = 0
        self.reset()

    def reset(self) -> None:
        """reset analyser, clear any state"""
        self._done = False
        self._total_chars = 0
        self._freq_chars = 0

    def feed(self, char: Union[bytes, bytearray], char_len: int) -> None:
        """feed a character with known length"""
        if char_len == 2:
            order = self.get_order(char)
        else:
            order = -1
        if order >= 0:
            self._total_chars += 1
            if order < self._table_size:
                if 512 > self._char_to_freq_order[order]:
                    self._freq_chars += 1

    def get_confidence(self) -> float:
        """return confidence based on existing data"""
        if self._total_chars <= 0 or self._freq_chars <= self.MINIMUM_DATA_THRESHOLD:
            return self.SURE_NO
        if self._total_chars != self._freq_chars:
            r = self._freq_chars / ((self._total_chars - self._freq_chars) * self.typical_distribution_ratio)
            if r < self.SURE_YES:
                return r
        return self.SURE_YES

    def got_enough_data(self) -> bool:
        return self._total_chars > self.ENOUGH_DATA_THRESHOLD

    def get_order(self, _: Union[bytes, bytearray]) -> int:
        return -1