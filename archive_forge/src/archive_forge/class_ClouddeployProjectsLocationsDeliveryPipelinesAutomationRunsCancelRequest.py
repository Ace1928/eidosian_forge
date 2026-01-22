from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsCancelRequest(_messages.Message):
    """A
  ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsCancelRequest
  object.

  Fields:
    cancelAutomationRunRequest: A CancelAutomationRunRequest resource to be
      passed as the request body.
    name: Required. Name of the `AutomationRun`. Format is `projects/{project}
      /locations/{location}/deliveryPipelines/{delivery_pipeline}/automationRu
      ns/{automation_run}`.
  """
    cancelAutomationRunRequest = _messages.MessageField('CancelAutomationRunRequest', 1)
    name = _messages.StringField(2, required=True)