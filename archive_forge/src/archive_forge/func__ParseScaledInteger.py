from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def _ParseScaledInteger(units, string, default_unit='', type_abbr='B'):
    """Parses and returns a units scaled integer from string.

  ISO/IEC/SI rules relaxed to ignore case in unit and type names/abbreviations.

  Args:
    units: {str: int} map of unit prefix => size.
    string: The string to parse the integer + units.
    default_unit: The default unit prefix name.
    type_abbr: The optional type abbreviation suffix, validated but otherwise
      ignored.

  Raises:
    ValueError: on invalid input.

  Returns:
    The scaled integer value.
  """
    match = re.match(_INTEGER_SUFFIX_TYPE_PATTERN, string, re.VERBOSE)
    if not match:
        optional_type_abbr = '[' + type_abbr + ']' if type_abbr else ''
        raise ValueError('[{}] must the form INTEGER[UNIT]{} where units may be one of [{}].'.format(string, optional_type_abbr, ','.join(_UnitsByMagnitude(units, type_abbr))))
    suffix = match.group('suffix') or ''
    size = GetUnitSize(suffix, type_abbr=type_abbr, default_unit=default_unit, units=units)
    amount = int(match.group('amount'))
    return amount * size