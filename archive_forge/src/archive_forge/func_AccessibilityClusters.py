from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
def AccessibilityClusters(self, request, global_params=None):
    """Lists accessibility clusters for a given Step May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if an argument in the request happens to be invalid; e.g. if the locale format is incorrect - NOT_FOUND - if the containing Step does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsAccessibilityClustersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStepAccessibilityClustersResponse) The response message.
      """
    config = self.GetMethodConfig('AccessibilityClusters')
    return self._RunMethod(config, request, global_params=global_params)