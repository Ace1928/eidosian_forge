import stringprep
from typing import Callable, Tuple
import unicodedata
An implementation of RFC4013 SASLprep.
    :param data:
        The string to SASLprep.
    :param prohibit_unassigned_code_points:
        RFC 3454 and RFCs for various SASL mechanisms distinguish between
        `queries` (unassigned code points allowed) and
        `stored strings` (unassigned code points prohibited). Defaults
        to ``True`` (unassigned code points are prohibited).
    :return: The SASLprep'ed version of `data`.
    