from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsGetRequest
  object.

  Fields:
    name: Required. Name of the `AutomationRun`. Format must be `projects/{pro
      ject}/locations/{location}/deliveryPipelines/{delivery_pipeline}/automat
      ionRuns/{automation_run}`.
  """
    name = _messages.StringField(1, required=True)