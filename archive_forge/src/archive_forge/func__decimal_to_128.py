from __future__ import annotations
import decimal
import struct
from typing import Any, Sequence, Tuple, Type, Union
def _decimal_to_128(value: _VALUE_OPTIONS) -> Tuple[int, int]:
    """Converts a decimal.Decimal to BID (high bits, low bits).

    :Parameters:
      - `value`: An instance of decimal.Decimal
    """
    with decimal.localcontext(_DEC128_CTX) as ctx:
        value = ctx.create_decimal(value)
    if value.is_infinite():
        return _NINF if value.is_signed() else _PINF
    sign, digits, exponent = value.as_tuple()
    if value.is_nan():
        if digits:
            raise ValueError('NaN with debug payload is not supported')
        if value.is_snan():
            return _NSNAN if value.is_signed() else _PSNAN
        return _NNAN if value.is_signed() else _PNAN
    significand = int(''.join([str(digit) for digit in digits]))
    bit_length = significand.bit_length()
    high = 0
    low = 0
    for i in range(min(64, bit_length)):
        if significand & 1 << i:
            low |= 1 << i
    for i in range(64, bit_length):
        if significand & 1 << i:
            high |= 1 << i - 64
    biased_exponent = exponent + _EXPONENT_BIAS
    if high >> 49 == 1:
        high = high & 140737488355327
        high |= _EXPONENT_MASK
        high |= (biased_exponent & 16383) << 47
    else:
        high |= biased_exponent << 49
    if sign:
        high |= _SIGN
    return (high, low)