from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportFormatValueValuesEnum(_messages.Enum):
    """Required. The format in which the Scope report bytes should be
    returned.

    Values:
      AUDIT_SCOPE_REPORT_FORMAT_UNSPECIFIED: Unspecified. Invalid format.
      AUDIT_SCOPE_REPORT_FORMAT_ODF: Audit Scope Report creation format is
        Open Document.
    """
    AUDIT_SCOPE_REPORT_FORMAT_UNSPECIFIED = 0
    AUDIT_SCOPE_REPORT_FORMAT_ODF = 1