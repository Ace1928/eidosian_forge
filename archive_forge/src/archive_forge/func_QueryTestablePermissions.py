from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
def QueryTestablePermissions(self, request, global_params=None):
    """Lists every permission that you can test on a resource. A permission is testable if you can check whether a principal has that permission on the resource.

      Args:
        request: (QueryTestablePermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryTestablePermissionsResponse) The response message.
      """
    config = self.GetMethodConfig('QueryTestablePermissions')
    return self._RunMethod(config, request, global_params=global_params)