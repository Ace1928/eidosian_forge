from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.publicca.v1beta1 import publicca_v1beta1_messages as messages
Creates a new ExternalAccountKey bound to the project.

      Args:
        request: (PubliccaProjectsLocationsExternalAccountKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExternalAccountKey) The response message.
      