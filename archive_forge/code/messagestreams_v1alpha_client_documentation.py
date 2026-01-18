from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.messagestreams.v1alpha import messagestreams_v1alpha_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (MessagestreamsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      