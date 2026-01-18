from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
Export generated customer metadata for a given resource.

      Args:
        request: (RunProjectsLocationsExportMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2Metadata) The response message.
      