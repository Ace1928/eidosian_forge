from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
Tests a value against the result of merging consumer policies in the resource hierarchy. This operation is designed to be used for building policy-aware UIs and command-line tools, not for access checking.

      Args:
        request: (ServiceusageTestEnabledRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (State) The response message.
      