from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha2 import gkehub_v1alpha2_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (GkehubProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      