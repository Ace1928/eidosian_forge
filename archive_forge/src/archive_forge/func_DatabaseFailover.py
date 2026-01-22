from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def DatabaseFailover(self, request, global_params=None):
    """Triggers database failover (only for highly resilient environments).

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDatabaseFailoverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DatabaseFailover')
    return self._RunMethod(config, request, global_params=global_params)