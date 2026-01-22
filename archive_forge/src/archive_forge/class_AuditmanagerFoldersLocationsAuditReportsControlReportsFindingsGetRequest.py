from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerFoldersLocationsAuditReportsControlReportsFindingsGetRequest(_messages.Message):
    """A
  AuditmanagerFoldersLocationsAuditReportsControlReportsFindingsGetRequest
  object.

  Fields:
    name: Required. Format projects/{project-id}/locations/{location}/auditRep
      orts/{auditReportName}/controlReports/{controlId}/findings/{finding},
      folders/{folder-id}/locations/{location}/auditReports/{auditReportName}/
      controlReports/{controlId}/findings/{finding}
  """
    name = _messages.StringField(1, required=True)