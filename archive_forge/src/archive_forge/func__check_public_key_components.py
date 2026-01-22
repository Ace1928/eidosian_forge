from __future__ import annotations
import abc
import typing
from math import gcd
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def _check_public_key_components(e: int, n: int) -> None:
    if n < 3:
        raise ValueError('n must be >= 3.')
    if e < 3 or e >= n:
        raise ValueError('e must be >= 3 and < n.')
    if e & 1 == 0:
        raise ValueError('e must be odd.')