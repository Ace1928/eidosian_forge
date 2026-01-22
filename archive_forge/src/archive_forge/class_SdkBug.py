from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SdkBug(_messages.Message):
    """A bug found in the Dataflow SDK.

  Enums:
    SeverityValueValuesEnum: Output only. How severe the SDK bug is.
    TypeValueValuesEnum: Output only. Describes the impact of this SDK bug.

  Fields:
    severity: Output only. How severe the SDK bug is.
    type: Output only. Describes the impact of this SDK bug.
    uri: Output only. Link to more information on the bug.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """Output only. How severe the SDK bug is.

    Values:
      SEVERITY_UNSPECIFIED: A bug of unknown severity.
      NOTICE: A minor bug that that may reduce reliability or performance for
        some jobs. Impact will be minimal or non-existent for most jobs.
      WARNING: A bug that has some likelihood of causing performance
        degradation, data loss, or job failures.
      SEVERE: A bug with extremely significant impact. Jobs may fail
        erroneously, performance may be severely degraded, and data loss may
        be very likely.
    """
        SEVERITY_UNSPECIFIED = 0
        NOTICE = 1
        WARNING = 2
        SEVERE = 3

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Describes the impact of this SDK bug.

    Values:
      TYPE_UNSPECIFIED: Unknown issue with this SDK.
      GENERAL: Catch-all for SDK bugs that don't fit in the below categories.
      PERFORMANCE: Using this version of the SDK may result in degraded
        performance.
      DATALOSS: Using this version of the SDK may cause data loss.
    """
        TYPE_UNSPECIFIED = 0
        GENERAL = 1
        PERFORMANCE = 2
        DATALOSS = 3
    severity = _messages.EnumField('SeverityValueValuesEnum', 1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)
    uri = _messages.StringField(3)