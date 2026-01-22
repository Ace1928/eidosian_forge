from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesGetRequest object.

  Fields:
    name: Required. Name of the `DeliveryPipeline`. Format must be `projects/{
      project_id}/locations/{location_name}/deliveryPipelines/{pipeline_name}`
      .
  """
    name = _messages.StringField(1, required=True)