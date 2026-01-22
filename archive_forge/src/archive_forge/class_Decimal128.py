from __future__ import annotations
import decimal
import struct
from typing import Any, Sequence, Tuple, Type, Union
class Decimal128:
    """BSON Decimal128 type::

      >>> Decimal128(Decimal("0.0005"))
      Decimal128('0.0005')
      >>> Decimal128("0.0005")
      Decimal128('0.0005')
      >>> Decimal128((3474527112516337664, 5))
      Decimal128('0.0005')

    :Parameters:
      - `value`: An instance of :class:`decimal.Decimal`, string, or tuple of
        (high bits, low bits) from Binary Integer Decimal (BID) format.

    .. note:: :class:`~Decimal128` uses an instance of :class:`decimal.Context`
      configured for IEEE-754 Decimal128 when validating parameters.
      Signals like :class:`decimal.InvalidOperation`, :class:`decimal.Inexact`,
      and :class:`decimal.Overflow` are trapped and raised as exceptions::

        >>> Decimal128(".13.1")
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          ...
        decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
        >>>
        >>> Decimal128("1E-6177")
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          ...
        decimal.Inexact: [<class 'decimal.Inexact'>]
        >>>
        >>> Decimal128("1E6145")
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          ...
        decimal.Overflow: [<class 'decimal.Overflow'>, <class 'decimal.Rounded'>]

      To ensure the result of a calculation can always be stored as BSON
      Decimal128 use the context returned by
      :func:`create_decimal128_context`::

        >>> import decimal
        >>> decimal128_ctx = create_decimal128_context()
        >>> with decimal.localcontext(decimal128_ctx) as ctx:
        ...     Decimal128(ctx.create_decimal(".13.3"))
        ...
        Decimal128('NaN')
        >>>
        >>> with decimal.localcontext(decimal128_ctx) as ctx:
        ...     Decimal128(ctx.create_decimal("1E-6177"))
        ...
        Decimal128('0E-6176')
        >>>
        >>> with decimal.localcontext(DECIMAL128_CTX) as ctx:
        ...     Decimal128(ctx.create_decimal("1E6145"))
        ...
        Decimal128('Infinity')

      To match the behavior of MongoDB's Decimal128 implementation
      str(Decimal(value)) may not match str(Decimal128(value)) for NaN values::

        >>> Decimal128(Decimal('NaN'))
        Decimal128('NaN')
        >>> Decimal128(Decimal('-NaN'))
        Decimal128('NaN')
        >>> Decimal128(Decimal('sNaN'))
        Decimal128('NaN')
        >>> Decimal128(Decimal('-sNaN'))
        Decimal128('NaN')

      However, :meth:`~Decimal128.to_decimal` will return the exact value::

        >>> Decimal128(Decimal('NaN')).to_decimal()
        Decimal('NaN')
        >>> Decimal128(Decimal('-NaN')).to_decimal()
        Decimal('-NaN')
        >>> Decimal128(Decimal('sNaN')).to_decimal()
        Decimal('sNaN')
        >>> Decimal128(Decimal('-sNaN')).to_decimal()
        Decimal('-sNaN')

      Two instances of :class:`Decimal128` compare equal if their Binary
      Integer Decimal encodings are equal::

        >>> Decimal128('NaN') == Decimal128('NaN')
        True
        >>> Decimal128('NaN').bid == Decimal128('NaN').bid
        True

      This differs from :class:`decimal.Decimal` comparisons for NaN::

        >>> Decimal('NaN') == Decimal('NaN')
        False
    """
    __slots__ = ('__high', '__low')
    _type_marker = 19

    def __init__(self, value: _VALUE_OPTIONS) -> None:
        if isinstance(value, (str, decimal.Decimal)):
            self.__high, self.__low = _decimal_to_128(value)
        elif isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError('Invalid size for creation of Decimal128 from list or tuple. Must have exactly 2 elements.')
            self.__high, self.__low = value
        else:
            raise TypeError(f'Cannot convert {value!r} to Decimal128')

    def to_decimal(self) -> decimal.Decimal:
        """Returns an instance of :class:`decimal.Decimal` for this
        :class:`Decimal128`.
        """
        high = self.__high
        low = self.__low
        sign = 1 if high & _SIGN else 0
        if high & _SNAN == _SNAN:
            return decimal.Decimal((sign, (), 'N'))
        elif high & _NAN == _NAN:
            return decimal.Decimal((sign, (), 'n'))
        elif high & _INF == _INF:
            return decimal.Decimal((sign, (), 'F'))
        if high & _EXPONENT_MASK == _EXPONENT_MASK:
            exponent = ((high & 2305807824841605120) >> 47) - _EXPONENT_BIAS
            return decimal.Decimal((sign, (0,), exponent))
        else:
            exponent = ((high & 9223231299366420480) >> 49) - _EXPONENT_BIAS
        arr = bytearray(15)
        mask = 255
        for i in range(14, 6, -1):
            arr[i] = (low & mask) >> (14 - i << 3)
            mask = mask << 8
        mask = 255
        for i in range(6, 0, -1):
            arr[i] = (high & mask) >> (6 - i << 3)
            mask = mask << 8
        mask = 281474976710656
        arr[0] = (high & mask) >> 48
        digits = tuple((int(digit) for digit in str(int.from_bytes(arr, 'big'))))
        with decimal.localcontext(_DEC128_CTX) as ctx:
            return ctx.create_decimal((sign, digits, exponent))

    @classmethod
    def from_bid(cls: Type[Decimal128], value: bytes) -> Decimal128:
        """Create an instance of :class:`Decimal128` from Binary Integer
        Decimal string.

        :Parameters:
          - `value`: 16 byte string (128-bit IEEE 754-2008 decimal floating
            point in Binary Integer Decimal (BID) format).
        """
        if not isinstance(value, bytes):
            raise TypeError('value must be an instance of bytes')
        if len(value) != 16:
            raise ValueError('value must be exactly 16 bytes')
        return cls((_UNPACK_64(value[8:])[0], _UNPACK_64(value[:8])[0]))

    @property
    def bid(self) -> bytes:
        """The Binary Integer Decimal (BID) encoding of this instance."""
        return _PACK_64(self.__low) + _PACK_64(self.__high)

    def __str__(self) -> str:
        dec = self.to_decimal()
        if dec.is_nan():
            return 'NaN'
        return str(dec)

    def __repr__(self) -> str:
        return f"Decimal128('{self!s}')"

    def __setstate__(self, value: Tuple[int, int]) -> None:
        self.__high, self.__low = value

    def __getstate__(self) -> Tuple[int, int]:
        return (self.__high, self.__low)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Decimal128):
            return self.bid == other.bid
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other