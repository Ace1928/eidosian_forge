from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
def FetchDatabaseProperties(self, request, global_params=None):
    """Fetches database properties.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsFetchDatabasePropertiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchDatabasePropertiesResponse) The response message.
      """
    config = self.GetMethodConfig('FetchDatabaseProperties')
    return self._RunMethod(config, request, global_params=global_params)