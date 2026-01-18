from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as messages
Get a list of patch jobs.

      Args:
        request: (OsconfigProjectsPatchJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPatchJobsResponse) The response message.
      