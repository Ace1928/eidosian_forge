from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAuditReportsResponse(_messages.Message):
    """Response message with all the audit reports.

  Fields:
    auditReports: Output only. The audit reports.
    nextPageToken: Output only. The token to retrieve the next page of
      results.
  """
    auditReports = _messages.MessageField('AuditReport', 1, repeated=True)
    nextPageToken = _messages.StringField(2)