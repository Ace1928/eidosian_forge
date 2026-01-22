from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def Subscribe(self, request, global_params=None):
    """Creates a subscription for the environment's Pub/Sub topic. The server will assign a random name for this subscription. The "name" and "push_config" must *not* be specified.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSubscribeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Subscription) The response message.
      """
    config = self.GetMethodConfig('Subscribe')
    return self._RunMethod(config, request, global_params=global_params)