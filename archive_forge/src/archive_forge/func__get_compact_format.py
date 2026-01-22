from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _get_compact_format(number: float | decimal.Decimal | str, compact_format: LocaleDataDict, locale: Locale, fraction_digits: int) -> tuple[decimal.Decimal, NumberPattern | None]:
    """Returns the number after dividing by the unit and the format pattern to use.
    The algorithm is described here:
    https://www.unicode.org/reports/tr35/tr35-45/tr35-numbers.html#Compact_Number_Formats.
    """
    if not isinstance(number, decimal.Decimal):
        number = decimal.Decimal(str(number))
    if number.is_nan() or number.is_infinite():
        return (number, None)
    format = None
    for magnitude in sorted([int(m) for m in compact_format['other']], reverse=True):
        if abs(number) >= magnitude:
            format = compact_format['other'][str(magnitude)]
            pattern = parse_pattern(format).pattern
            if pattern == '0':
                break
            number = cast(decimal.Decimal, number / (magnitude // 10 ** (pattern.count('0') - 1)))
            rounded = round(number, fraction_digits)
            plural_form = locale.plural_form(abs(number))
            if plural_form not in compact_format:
                plural_form = 'other'
            if number == 1 and '1' in compact_format:
                plural_form = '1'
            format = compact_format[plural_form][str(magnitude)]
            number = rounded
            break
    return (number, format)