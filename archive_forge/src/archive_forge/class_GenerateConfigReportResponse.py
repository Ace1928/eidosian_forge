from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateConfigReportResponse(_messages.Message):
    """Response message for GenerateConfigReport method.

  Fields:
    changeReports: list of ChangeReport, each corresponding to comparison
      between two service configurations.
    diagnostics: Errors / Linter warnings associated with the service
      definition this report belongs to.
    id: ID of the service configuration this report belongs to.
    serviceName: Name of the service this report belongs to.
  """
    changeReports = _messages.MessageField('ChangeReport', 1, repeated=True)
    diagnostics = _messages.MessageField('Diagnostic', 2, repeated=True)
    id = _messages.StringField(3)
    serviceName = _messages.StringField(4)