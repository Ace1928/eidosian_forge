from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def GetBinaryUnitSize(suffix, type_abbr='B', default_unit=''):
    """Returns the binary size per unit for binary suffix string.

  Args:
    suffix: str, A case insensitive unit suffix string with optional type
      abbreviation.
    type_abbr: str, The optional case insensitive type abbreviation following
      the suffix.
    default_unit: The default unit prefix name.

  Raises:
    ValueError for unknown units.

  Returns:
    The binary size per unit for a unit+type_abbr suffix.
  """
    return GetUnitSize(suffix, type_abbr=type_abbr, default_unit=default_unit, units=_BINARY_UNITS)