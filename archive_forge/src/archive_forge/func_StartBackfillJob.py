from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastream.v1 import datastream_v1_messages as messages
def StartBackfillJob(self, request, global_params=None):
    """Use this method to start a backfill job for the specified stream object.

      Args:
        request: (DatastreamProjectsLocationsStreamsObjectsStartBackfillJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StartBackfillJobResponse) The response message.
      """
    config = self.GetMethodConfig('StartBackfillJob')
    return self._RunMethod(config, request, global_params=global_params)