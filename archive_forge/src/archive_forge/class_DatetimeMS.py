from __future__ import annotations
import calendar
import datetime
import functools
from typing import Any, Union, cast
from bson.codec_options import DEFAULT_CODEC_OPTIONS, CodecOptions, DatetimeConversion
from bson.errors import InvalidBSON
from bson.tz_util import utc
class DatetimeMS:
    """Represents a BSON UTC datetime."""
    __slots__ = ('_value',)

    def __init__(self, value: Union[int, datetime.datetime]):
        """Represents a BSON UTC datetime.

        BSON UTC datetimes are defined as an int64 of milliseconds since the
        Unix epoch. The principal use of DatetimeMS is to represent
        datetimes outside the range of the Python builtin
        :class:`~datetime.datetime` class when
        encoding/decoding BSON.

        To decode UTC datetimes as a ``DatetimeMS``, `datetime_conversion` in
        :class:`~bson.CodecOptions` must be set to 'datetime_ms' or
        'datetime_auto'. See :ref:`handling-out-of-range-datetimes` for
        details.

        :Parameters:
          - `value`: An instance of :class:`datetime.datetime` to be
            represented as milliseconds since the Unix epoch, or int of
            milliseconds since the Unix epoch.
        """
        if isinstance(value, int):
            if not -2 ** 63 <= value <= 2 ** 63 - 1:
                raise OverflowError('Must be a 64-bit integer of milliseconds')
            self._value = value
        elif isinstance(value, datetime.datetime):
            self._value = _datetime_to_millis(value)
        else:
            raise TypeError(f'{type(value)} is not a valid type for DatetimeMS')

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return type(self).__name__ + '(' + str(self._value) + ')'

    def __lt__(self, other: Union[DatetimeMS, int]) -> bool:
        return self._value < other

    def __le__(self, other: Union[DatetimeMS, int]) -> bool:
        return self._value <= other

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DatetimeMS):
            return self._value == other._value
        return False

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, DatetimeMS):
            return self._value != other._value
        return True

    def __gt__(self, other: Union[DatetimeMS, int]) -> bool:
        return self._value > other

    def __ge__(self, other: Union[DatetimeMS, int]) -> bool:
        return self._value >= other
    _type_marker = 9

    def as_datetime(self, codec_options: CodecOptions[Any]=DEFAULT_CODEC_OPTIONS) -> datetime.datetime:
        """Create a Python :class:`~datetime.datetime` from this DatetimeMS object.

        :Parameters:
          - `codec_options`: A CodecOptions instance for specifying how the
            resulting DatetimeMS object will be formatted using ``tz_aware``
            and ``tz_info``. Defaults to
            :const:`~bson.codec_options.DEFAULT_CODEC_OPTIONS`.
        """
        return cast(datetime.datetime, _millis_to_datetime(self._value, codec_options))

    def __int__(self) -> int:
        return self._value