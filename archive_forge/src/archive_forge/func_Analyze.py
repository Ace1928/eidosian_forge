from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def Analyze(self, request, global_params=None):
    """Analyze a Batch for possible recommendations and insights.

      Args:
        request: (DataprocProjectsLocationsBatchesAnalyzeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Analyze')
    return self._RunMethod(config, request, global_params=global_params)