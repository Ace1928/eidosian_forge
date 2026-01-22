from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportSummary(_messages.Message):
    """Represents additional information for an audit operation.

  Fields:
    compliantCount: Number of compliant checks.
    errorCount: Number of checks that could not be performed due to errors.
    totalCount: Total number of checks.
    unknownsCount: Number of checks with unknown status.
    violationCount: Number of checks with violations.
  """
    compliantCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    errorCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    totalCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    unknownsCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    violationCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)