from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SignalClassValueValuesEnum(_messages.Enum):
    """Required. The class of the signal, such as if it's a THREAT or
    VULNERABILITY.

    Values:
      CLASS_UNSPECIFIED: Unspecified signal class.
      THREAT: Describes unwanted or malicious activity.
      VULNERABILITY: Describes a potential weakness in software that increases
        risk to Confidentiality & Integrity & Availability.
      MISCONFIGURATION: Describes a potential weakness in cloud resource/asset
        configuration that increases risk.
      OBSERVATION: Describes a security observation that is for informational
        purposes.
      ERROR: Describes an error that prevents some SCC functionality.
    """
    CLASS_UNSPECIFIED = 0
    THREAT = 1
    VULNERABILITY = 2
    MISCONFIGURATION = 3
    OBSERVATION = 4
    ERROR = 5