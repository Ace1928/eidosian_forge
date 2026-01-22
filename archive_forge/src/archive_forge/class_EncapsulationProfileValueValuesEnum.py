from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncapsulationProfileValueValuesEnum(_messages.Enum):
    """Specify the encapsulation protocol and what metadata to include in
    incoming encapsulated packet headers.

    Values:
      GENEVE_SECURITY_V1: Use GENEVE encapsulation protocol and include the
        SECURITY_V1 set of GENEVE headers.
      UNSPECIFIED_ENCAPSULATION_PROFILE: <no description>
    """
    GENEVE_SECURITY_V1 = 0
    UNSPECIFIED_ENCAPSULATION_PROFILE = 1