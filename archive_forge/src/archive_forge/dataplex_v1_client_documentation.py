from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
Searches for entries matching given query and scope.

      Args:
        request: (DataplexProjectsLocationsSearchEntriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1SearchEntriesResponse) The response message.
      