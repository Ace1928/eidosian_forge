from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FixableTotalByDigest(_messages.Message):
    """Per resource and severity counts of fixable and total vulnerabilities.

  Enums:
    SeverityValueValuesEnum: The severity for this count. SEVERITY_UNSPECIFIED
      indicates total across all severities.

  Fields:
    fixableCount: The number of fixable vulnerabilities associated with this
      resource.
    resourceUri: The affected resource.
    severity: The severity for this count. SEVERITY_UNSPECIFIED indicates
      total across all severities.
    totalCount: The total number of vulnerabilities associated with this
      resource.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity for this count. SEVERITY_UNSPECIFIED indicates total
    across all severities.

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
    fixableCount = _messages.IntegerField(1)
    resourceUri = _messages.StringField(2)
    severity = _messages.EnumField('SeverityValueValuesEnum', 3)
    totalCount = _messages.IntegerField(4)