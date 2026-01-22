import itertools
from typing import Dict, Iterator, List, Optional, Sequence, cast
import numpy as np
def _octet_bits(o: int) -> List[int]:
    """
    Get the bits of an octet.

    :param o: The octets.
    :return: The bits as a list in LSB-to-MSB order.
    """
    if not isinstance(o, int):
        raise TypeError('o should be an int')
    if not 0 <= o <= 255:
        raise ValueError('o should be between 0 and 255 inclusive')
    bits = [0] * 8
    for i in range(8):
        if 1 == o & 1:
            bits[i] = 1
        o = o >> 1
    return bits