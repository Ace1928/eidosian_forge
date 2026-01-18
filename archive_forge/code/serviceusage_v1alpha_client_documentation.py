from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1alpha import serviceusage_v1alpha_messages as messages
Get service identity for service.

      Args:
        request: (ServiceusageServicesGetIdentityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceIdentity) The response message.
      