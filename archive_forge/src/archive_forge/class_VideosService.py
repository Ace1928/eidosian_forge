from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.videointelligence.v1 import videointelligence_v1_messages as messages
class VideosService(base_api.BaseApiService):
    """Service class for the videos resource."""
    _NAME = 'videos'

    def __init__(self, client):
        super(VideointelligenceV1.VideosService, self).__init__(client)
        self._upload_configs = {}

    def Annotate(self, request, global_params=None):
        """Performs asynchronous video annotation. Progress and results can be retrieved through the `google.longrunning.Operations` interface. `Operation.metadata` contains `AnnotateVideoProgress` (progress). `Operation.response` contains `AnnotateVideoResponse` (results).

      Args:
        request: (GoogleCloudVideointelligenceV1AnnotateVideoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Annotate')
        return self._RunMethod(config, request, global_params=global_params)
    Annotate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='videointelligence.videos.annotate', ordered_params=[], path_params=[], query_params=[], relative_path='v1/videos:annotate', request_field='<request>', request_type_name='GoogleCloudVideointelligenceV1AnnotateVideoRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)