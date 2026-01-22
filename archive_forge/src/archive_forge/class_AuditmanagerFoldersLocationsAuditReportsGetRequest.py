from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerFoldersLocationsAuditReportsGetRequest(_messages.Message):
    """A AuditmanagerFoldersLocationsAuditReportsGetRequest object.

  Fields:
    name: Required. Format projects/{project-
      id}/locations/{location}/auditReports/{auditReportName},
      folders/{folder-id}/locations/{location}/auditReports/{auditReportName}
  """
    name = _messages.StringField(1, required=True)