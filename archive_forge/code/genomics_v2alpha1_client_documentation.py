from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.genomics.v2alpha1 import genomics_v2alpha1_messages as messages
The worker uses this method to retrieve the assigned operation and provide periodic status updates.

      Args:
        request: (GenomicsWorkersCheckInRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckInResponse) The response message.
      