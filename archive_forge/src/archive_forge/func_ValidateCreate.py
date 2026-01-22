from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def ValidateCreate(self, request, global_params=None):
    """ValidateCreateMembership is a preflight check for CreateMembership. It checks the following: 1. Caller has the required `gkehub.memberships.create` permission. 2. The membership_id is still available.

      Args:
        request: (GkehubProjectsLocationsMembershipsValidateCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateCreateMembershipResponse) The response message.
      """
    config = self.GetMethodConfig('ValidateCreate')
    return self._RunMethod(config, request, global_params=global_params)