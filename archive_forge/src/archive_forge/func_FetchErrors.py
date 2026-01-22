from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastream.v1alpha1 import datastream_v1alpha1_messages as messages
def FetchErrors(self, request, global_params=None):
    """Use this method to fetch any errors associated with a stream.

      Args:
        request: (DatastreamProjectsLocationsStreamsFetchErrorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('FetchErrors')
    return self._RunMethod(config, request, global_params=global_params)