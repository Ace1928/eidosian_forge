from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkflowsRunRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkflowsRunRequest object.

  Fields:
    name: Required. Format:
      `projects/{project}/locations/{location}/workflow/{workflow}`
    runWorkflowRequest: A RunWorkflowRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    runWorkflowRequest = _messages.MessageField('RunWorkflowRequest', 2)