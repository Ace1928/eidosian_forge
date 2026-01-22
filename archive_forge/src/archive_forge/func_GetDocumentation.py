from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetDocumentation(self, request, global_params=None):
    """Gets the documentation for the specified catalog item.

      Args:
        request: (ApigeeOrganizationsSitesApidocsGetDocumentationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocDocumentationResponse) The response message.
      """
    config = self.GetMethodConfig('GetDocumentation')
    return self._RunMethod(config, request, global_params=global_params)