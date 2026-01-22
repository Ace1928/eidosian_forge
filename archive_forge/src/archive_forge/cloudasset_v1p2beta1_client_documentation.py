from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p2beta1 import cloudasset_v1p2beta1_messages as messages
Exports assets with time and resource types to a given Cloud Storage.
location. The output format is newline-delimited JSON.
This API implements the google.longrunning.Operation API allowing you
to keep track of the export.

      Args:
        request: (CloudassetExportAssetsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      