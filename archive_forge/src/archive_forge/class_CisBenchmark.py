from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CisBenchmark(_messages.Message):
    """A compliance check that is a CIS benchmark.

  Enums:
    SeverityValueValuesEnum:

  Fields:
    profileLevel: A integer attribute.
    severity: A SeverityValueValuesEnum attribute.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """SeverityValueValuesEnum enum type.

    Values:
      SEVERITY_UNSPECIFIED: Unknown.
      MINIMAL: Minimal severity.
      LOW: Low severity.
      MEDIUM: Medium severity.
      HIGH: High severity.
      CRITICAL: Critical severity.
    """
        SEVERITY_UNSPECIFIED = 0
        MINIMAL = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5
    profileLevel = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    severity = _messages.EnumField('SeverityValueValuesEnum', 2)