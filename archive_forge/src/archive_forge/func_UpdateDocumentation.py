from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateDocumentation(self, request, global_params=None):
    """Updates the documentation for the specified catalog item. Note that the documentation file contents will not be populated in the return message.

      Args:
        request: (ApigeeOrganizationsSitesApidocsUpdateDocumentationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiDocDocumentationResponse) The response message.
      """
    config = self.GetMethodConfig('UpdateDocumentation')
    return self._RunMethod(config, request, global_params=global_params)