from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsExportDataRequest(_messages.Message):
    """A
  WorkflowexecutionsProjectsLocationsWorkflowsExecutionsExportDataRequest
  object.

  Fields:
    name: Required. Name of the execution for which data is to be exported.
      Format: projects/{project}/locations/{location}/workflows/{workflow}/exe
      cutions/{execution}
  """
    name = _messages.StringField(1, required=True)