from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p5beta1 import cloudasset_v1p5beta1_messages as messages
Lists assets with time and resource types and returns paged results in.
response.

      Args:
        request: (CloudassetAssetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAssetsResponse) The response message.
      