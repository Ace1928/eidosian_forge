from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
class Latin1Prober(CharSetProber):

    def __init__(self) -> None:
        super().__init__()
        self._last_char_class = OTH
        self._freq_counter: List[int] = []
        self.reset()

    def reset(self) -> None:
        self._last_char_class = OTH
        self._freq_counter = [0] * FREQ_CAT_NUM
        super().reset()

    @property
    def charset_name(self) -> str:
        return 'ISO-8859-1'

    @property
    def language(self) -> str:
        return ''

    def feed(self, byte_str: Union[bytes, bytearray]) -> ProbingState:
        byte_str = self.remove_xml_tags(byte_str)
        for c in byte_str:
            char_class = Latin1_CharToClass[c]
            freq = Latin1ClassModel[self._last_char_class * CLASS_NUM + char_class]
            if freq == 0:
                self._state = ProbingState.NOT_ME
                break
            self._freq_counter[freq] += 1
            self._last_char_class = char_class
        return self.state

    def get_confidence(self) -> float:
        if self.state == ProbingState.NOT_ME:
            return 0.01
        total = sum(self._freq_counter)
        confidence = 0.0 if total < 0.01 else (self._freq_counter[3] - self._freq_counter[1] * 20.0) / total
        confidence = max(confidence, 0.0)
        confidence *= 0.73
        return confidence