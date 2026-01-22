from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datapipelines.v1 import datapipelines_v1_messages as messages
Freezes pipeline execution permanently. If there's a corresponding scheduler entry, it's deleted, and the pipeline state is changed to "ARCHIVED". However, pipeline metadata is retained.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1Pipeline) The response message.
      