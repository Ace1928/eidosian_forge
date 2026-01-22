from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesReportEventRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesReportEventRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    reportInstanceEventRequest: A ReportInstanceEventRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    reportInstanceEventRequest = _messages.MessageField('ReportInstanceEventRequest', 2)