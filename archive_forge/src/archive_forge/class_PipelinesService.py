from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.genomics.v2alpha1 import genomics_v2alpha1_messages as messages
class PipelinesService(base_api.BaseApiService):
    """Service class for the pipelines resource."""
    _NAME = 'pipelines'

    def __init__(self, client):
        super(GenomicsV2alpha1.PipelinesService, self).__init__(client)
        self._upload_configs = {}

    def Run(self, request, global_params=None):
        """Runs a pipeline. The returned Operation's metadata field will contain a google.genomics.v2alpha1.Metadata object describing the status of the pipeline execution. The [response] field will contain a google.genomics.v2alpha1.RunPipelineResponse object if the pipeline completes successfully. **Note:** Before you can use this method, the Genomics Service Agent must have access to your project. This is done automatically when the Cloud Genomics API is first enabled, but if you delete this permission, or if you enabled the Cloud Genomics API before the v2alpha1 API launch, you must disable and re-enable the API to grant the Genomics Service Agent the required permissions. Authorization requires the following [Google IAM](https://cloud.google.com/iam/) permission: * `genomics.operations.create` [1]: /genomics/gsa.

      Args:
        request: (RunPipelineRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='genomics.pipelines.run', ordered_params=[], path_params=[], query_params=[], relative_path='v2alpha1/pipelines:run', request_field='<request>', request_type_name='RunPipelineRequest', response_type_name='Operation', supports_download=False)