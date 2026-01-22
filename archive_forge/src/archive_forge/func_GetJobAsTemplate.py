from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def GetJobAsTemplate(self, request, global_params=None):
    """Exports the resource representation for a job in a project as a template that can be used as a SubmitJobRequest.

      Args:
        request: (DataprocProjectsRegionsJobsGetJobAsTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
    config = self.GetMethodConfig('GetJobAsTemplate')
    return self._RunMethod(config, request, global_params=global_params)