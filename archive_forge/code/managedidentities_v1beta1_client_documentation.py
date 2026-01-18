from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1beta1 import managedidentities_v1beta1_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (ManagedidentitiesProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      