from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
def ListPermitted(self, request, global_params=None):
    """Lists permitted Scopes.

      Args:
        request: (GkehubProjectsLocationsScopesListPermittedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPermittedScopesResponse) The response message.
      """
    config = self.GetMethodConfig('ListPermitted')
    return self._RunMethod(config, request, global_params=global_params)