from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1alpha1 import cloudidentity_v1alpha1_messages as messages
Searches for `Group` resources matching a specified query.

      Args:
        request: (CloudidentityGroupsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchGroupsResponse) The response message.
      