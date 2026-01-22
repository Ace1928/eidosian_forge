from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerFoldersLocationsAuditReportsControlReportsGetRequest(_messages.Message):
    """A AuditmanagerFoldersLocationsAuditReportsControlReportsGetRequest
  object.

  Fields:
    name: Required. Format projects/{project-id}/locations/{location}/auditRep
      orts/{auditReportName}/controlReports/{controlId}, folders/{folder-id}/l
      ocations/{location}/auditReports/{auditReportName}/controlReports/{contr
      olId}
  """
    name = _messages.StringField(1, required=True)