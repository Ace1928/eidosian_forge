from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCancelRequest(_messages.Message):
    """A
  ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCancelRequest
  object.

  Fields:
    cancelRolloutRequest: A CancelRolloutRequest resource to be passed as the
      request body.
    name: Required. Name of the Rollout. Format is `projects/{project}/locatio
      ns/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}/ro
      llouts/{rollout}`.
  """
    cancelRolloutRequest = _messages.MessageField('CancelRolloutRequest', 1)
    name = _messages.StringField(2, required=True)