from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigeeregistry.v1 import apigeeregistry_v1_messages as messages
def TagRevision(self, request, global_params=None):
    """Adds a tag to a specified revision of a spec.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsTagRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
    config = self.GetMethodConfig('TagRevision')
    return self._RunMethod(config, request, global_params=global_params)