from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def GetUnitSize(suffix, type_abbr='B', default_unit='', units=None):
    """Returns the size per unit for binary suffix string.

  Args:
    suffix: str, A case insensitive unit suffix string with optional type
      abbreviation.
    type_abbr: str, The optional case insensitive type abbreviation following
      the suffix.
    default_unit: The default unit prefix name.
    units: {str: int} map of unit prefix => size.

  Raises:
    ValueError: on unknown units of type suffix.

  Returns:
    The binary size per unit for a unit+type_abbr suffix.
  """
    prefix = DeleteTypeAbbr(suffix, type_abbr)
    if not prefix:
        unit = default_unit
        if not unit:
            unit = ''
        elif unit.startswith('K'):
            unit = 'k' + unit[1:]
    else:
        unit = prefix[0].upper()
        if unit == 'K':
            unit = 'k'
        if len(prefix) > 1 and prefix[1] in ('i', 'I'):
            unit += 'i'
            prefix = prefix[2:]
        else:
            prefix = prefix[1:]
        if prefix:
            raise ValueError('Invalid type [{}] in [{}], expected [{}] or nothing.'.format(prefix, suffix, type_abbr))
    size = (units or _ISO_IEC_UNITS).get(unit)
    if not size:
        raise ValueError('Invalid suffix [{}] in [{}], expected one of [{}].'.format(unit, suffix, ','.join(_UnitsByMagnitude(units, ''))))
    return size