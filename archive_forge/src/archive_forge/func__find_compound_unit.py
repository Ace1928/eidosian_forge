from __future__ import annotations
import decimal
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.numbers import LC_NUMERIC, format_decimal
def _find_compound_unit(numerator_unit: str, denominator_unit: str, locale: Locale | str | None=LC_NUMERIC) -> str | None:
    """
    Find a predefined compound unit pattern.

    Used internally by format_compound_unit.

    >>> _find_compound_unit("kilometer", "hour", locale="en")
    'speed-kilometer-per-hour'

    >>> _find_compound_unit("mile", "gallon", locale="en")
    'consumption-mile-per-gallon'

    If no predefined compound pattern can be found, `None` is returned.

    >>> _find_compound_unit("gallon", "mile", locale="en")

    >>> _find_compound_unit("horse", "purple", locale="en")

    :param numerator_unit: The numerator unit's identifier
    :param denominator_unit: The denominator unit's identifier
    :param locale: the `Locale` object or locale identifier
    :return: A key to the `unit_patterns` mapping, or None.
    :rtype: str|None
    """
    locale = Locale.parse(locale)
    resolved_numerator_unit = _find_unit_pattern(numerator_unit, locale=locale)
    resolved_denominator_unit = _find_unit_pattern(denominator_unit, locale=locale)
    if not (resolved_numerator_unit and resolved_denominator_unit):
        return None
    bare_numerator_unit = resolved_numerator_unit.split('-', 1)[-1]
    bare_denominator_unit = resolved_denominator_unit.split('-', 1)[-1]
    return _find_unit_pattern(f'{bare_numerator_unit}-per-{bare_denominator_unit}', locale=locale)