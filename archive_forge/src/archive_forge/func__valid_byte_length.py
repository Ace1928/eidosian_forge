from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import (
from cryptography.hazmat.primitives import (
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
@staticmethod
def _valid_byte_length(value: int) -> bool:
    if not isinstance(value, int):
        raise TypeError('value must be of type int')
    value_bin = utils.int_to_bytes(1, value)
    if not 1 <= len(value_bin) <= 4:
        return False
    return True