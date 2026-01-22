from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerFoldersLocationsAuditScopeReportsGenerateRequest(_messages.Message):
    """A AuditmanagerFoldersLocationsAuditScopeReportsGenerateRequest object.

  Fields:
    generateAuditScopeReportRequest: A GenerateAuditScopeReportRequest
      resource to be passed as the request body.
    scope: Required. Scope for which the AuditScopeReport is required. Must be
      of format resource_type/resource_identifier Eg: projects/{project-
      id}/locations/{location}, folders/{folder-id}/locations/{location}
  """
    generateAuditScopeReportRequest = _messages.MessageField('GenerateAuditScopeReportRequest', 1)
    scope = _messages.StringField(2, required=True)