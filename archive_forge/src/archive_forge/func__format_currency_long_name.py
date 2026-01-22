from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _format_currency_long_name(number: float | decimal.Decimal | str, currency: str, format: str | NumberPattern | None=None, locale: Locale | str | None=LC_NUMERIC, currency_digits: bool=True, format_type: Literal['name', 'standard', 'accounting']='standard', decimal_quantization: bool=True, group_separator: bool=True, *, numbering_system: Literal['default'] | str='latn') -> str:
    locale = Locale.parse(locale)
    number_n = float(number) if isinstance(number, str) else number
    unit_pattern = get_currency_unit_pattern(currency, count=number_n, locale=locale)
    display_name = get_currency_name(currency, count=number_n, locale=locale)
    if not format:
        format = locale.decimal_formats[None]
    pattern = parse_pattern(format)
    number_part = pattern.apply(number, locale, currency=currency, currency_digits=currency_digits, decimal_quantization=decimal_quantization, group_separator=group_separator, numbering_system=numbering_system)
    return unit_pattern.format(number_part, display_name)