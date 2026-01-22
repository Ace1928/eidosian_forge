from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesReleasesGetRequest
  object.

  Fields:
    name: Required. Name of the `Release`. Format must be `projects/{project_i
      d}/locations/{location_name}/deliveryPipelines/{pipeline_name}/releases/
      {release_name}`.
  """
    name = _messages.StringField(1, required=True)