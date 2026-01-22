from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectgateway.v1alpha1 import connectgateway_v1alpha1_messages as messages
GenerateCredentials provides connection information that allows a user to access the specified membership using Connect Gateway.

      Args:
        request: (ConnectgatewayProjectsLocationsMembershipsGenerateCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateCredentialsResponse) The response message.
      