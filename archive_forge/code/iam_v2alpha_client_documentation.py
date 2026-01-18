from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v2alpha import iam_v2alpha_messages as messages
Updates the specified existing policy. Only `Policy.rules` and `Policy.display_name` may be updated. Need to provide 'Policy.etag' to enforce update from last read for optimistic concurrency control.

      Args:
        request: (GoogleIamV2alphaPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      