from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.file.v1p1alpha1 import file_v1p1alpha1_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (FileProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      