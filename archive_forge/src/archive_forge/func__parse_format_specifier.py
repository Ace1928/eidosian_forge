import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _parse_format_specifier(format_spec, _localeconv=None):
    """Parse and validate a format specifier.

    Turns a standard numeric format specifier into a dict, with the
    following entries:

      fill: fill character to pad field to minimum width
      align: alignment type, either '<', '>', '=' or '^'
      sign: either '+', '-' or ' '
      minimumwidth: nonnegative integer giving minimum width
      zeropad: boolean, indicating whether to pad with zeros
      thousands_sep: string to use as thousands separator, or ''
      grouping: grouping for thousands separators, in format
        used by localeconv
      decimal_point: string to use for decimal point
      precision: nonnegative integer giving precision, or None
      type: one of the characters 'eEfFgG%', or None

    """
    m = _parse_format_specifier_regex.match(format_spec)
    if m is None:
        raise ValueError('Invalid format specifier: ' + format_spec)
    format_dict = m.groupdict()
    fill = format_dict['fill']
    align = format_dict['align']
    format_dict['zeropad'] = format_dict['zeropad'] is not None
    if format_dict['zeropad']:
        if fill is not None:
            raise ValueError("Fill character conflicts with '0' in format specifier: " + format_spec)
        if align is not None:
            raise ValueError("Alignment conflicts with '0' in format specifier: " + format_spec)
    format_dict['fill'] = fill or ' '
    format_dict['align'] = align or '>'
    if format_dict['sign'] is None:
        format_dict['sign'] = '-'
    format_dict['minimumwidth'] = int(format_dict['minimumwidth'] or '0')
    if format_dict['precision'] is not None:
        format_dict['precision'] = int(format_dict['precision'])
    if format_dict['precision'] == 0:
        if format_dict['type'] is None or format_dict['type'] in 'gGn':
            format_dict['precision'] = 1
    if format_dict['type'] == 'n':
        format_dict['type'] = 'g'
        if _localeconv is None:
            _localeconv = _locale.localeconv()
        if format_dict['thousands_sep'] is not None:
            raise ValueError("Explicit thousands separator conflicts with 'n' type in format specifier: " + format_spec)
        format_dict['thousands_sep'] = _localeconv['thousands_sep']
        format_dict['grouping'] = _localeconv['grouping']
        format_dict['decimal_point'] = _localeconv['decimal_point']
    else:
        if format_dict['thousands_sep'] is None:
            format_dict['thousands_sep'] = ''
        format_dict['grouping'] = [3, 0]
        format_dict['decimal_point'] = '.'
    return format_dict