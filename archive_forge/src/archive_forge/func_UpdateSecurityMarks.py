from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
def UpdateSecurityMarks(self, request, global_params=None):
    """Updates security marks. For Finding Security marks, if no location is specified, finding is assumed to be in global. Assets Security Marks can only be accessed through global endpoint.

      Args:
        request: (SecuritycenterProjectsSourcesLocationsFindingsUpdateSecurityMarksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2SecurityMarks) The response message.
      """
    config = self.GetMethodConfig('UpdateSecurityMarks')
    return self._RunMethod(config, request, global_params=global_params)