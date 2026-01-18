from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runtimeconfig.v1beta1 import runtimeconfig_v1beta1_messages as messages
Updates a RuntimeConfig resource. The configuration must exist beforehand.

      Args:
        request: (RuntimeConfig) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RuntimeConfig) The response message.
      