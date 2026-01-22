from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def BatchImport(self, request, global_params=None):
    """Imports a list of externally generated EvaluatedAnnotations.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsSlicesBatchImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchImportEvaluatedAnnotationsResponse) The response message.
      """
    config = self.GetMethodConfig('BatchImport')
    return self._RunMethod(config, request, global_params=global_params)