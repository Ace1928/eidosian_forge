from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerFoldersLocationsAuditReportsGenerateRequest(_messages.Message):
    """A AuditmanagerFoldersLocationsAuditReportsGenerateRequest object.

  Fields:
    generateAuditReportRequest: A GenerateAuditReportRequest resource to be
      passed as the request body.
    scope: Required. Scope for which the AuditScopeReport is required. Must be
      of format resource_type/resource_identifier Eg: projects/{project-
      id}/locations/{location}, folders/{folder-id}/locations/{location}
  """
    generateAuditReportRequest = _messages.MessageField('GenerateAuditReportRequest', 1)
    scope = _messages.StringField(2, required=True)