from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeightTypeValueValuesEnum(_messages.Enum):
    """Specifies how the height is measured.

    Values:
      HEIGHT_TYPE_UNSPECIFIED: Unspecified height type.
      HEIGHT_TYPE_AGL: AGL height is measured relative to the ground level.
      HEIGHT_TYPE_AMSL: AMSL height is measured relative to the mean sea
        level.
    """
    HEIGHT_TYPE_UNSPECIFIED = 0
    HEIGHT_TYPE_AGL = 1
    HEIGHT_TYPE_AMSL = 2