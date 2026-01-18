from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
def validate_utf16_characters(self, pair: List[int]) -> None:
    """
        Validate if the pair of bytes is  valid UTF-16.

        UTF-16 is valid in the range 0x0000 - 0xFFFF excluding 0xD800 - 0xFFFF
        with an exception for surrogate pairs, which must be in the range
        0xD800-0xDBFF followed by 0xDC00-0xDFFF

        https://en.wikipedia.org/wiki/UTF-16
        """
    if not self.first_half_surrogate_pair_detected_16be:
        if 216 <= pair[0] <= 219:
            self.first_half_surrogate_pair_detected_16be = True
        elif 220 <= pair[0] <= 223:
            self.invalid_utf16be = True
    elif 220 <= pair[0] <= 223:
        self.first_half_surrogate_pair_detected_16be = False
    else:
        self.invalid_utf16be = True
    if not self.first_half_surrogate_pair_detected_16le:
        if 216 <= pair[1] <= 219:
            self.first_half_surrogate_pair_detected_16le = True
        elif 220 <= pair[1] <= 223:
            self.invalid_utf16le = True
    elif 220 <= pair[1] <= 223:
        self.first_half_surrogate_pair_detected_16le = False
    else:
        self.invalid_utf16le = True