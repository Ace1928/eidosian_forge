from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MechanismValueValuesEnum(_messages.Enum):
    """MechanismValueValuesEnum enum type.

    Values:
      AUTH_MECHANISM_UNSPECIFIED: <no description>
      PLAIN: <no description>
      SHA_256: <no description>
      SHA_512: <no description>
    """
    AUTH_MECHANISM_UNSPECIFIED = 0
    PLAIN = 1
    SHA_256 = 2
    SHA_512 = 3