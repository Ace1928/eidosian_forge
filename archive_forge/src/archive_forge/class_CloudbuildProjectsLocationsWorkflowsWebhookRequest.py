from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkflowsWebhookRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkflowsWebhookRequest object.

  Fields:
    processWorkflowTriggerWebhookRequest: A
      ProcessWorkflowTriggerWebhookRequest resource to be passed as the
      request body.
    workflow: Required. Format:
      `projects/{project}/locations/{location}/workflow/{workflow}`
  """
    processWorkflowTriggerWebhookRequest = _messages.MessageField('ProcessWorkflowTriggerWebhookRequest', 1)
    workflow = _messages.StringField(2, required=True)