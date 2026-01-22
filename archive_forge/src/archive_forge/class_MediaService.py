from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class MediaService(base_api.BaseApiService):
    """Service class for the media resource."""
    _NAME = 'media'

    def __init__(self, client):
        super(AiplatformV1beta1.MediaService, self).__init__(client)
        self._upload_configs = {'Upload': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=None, resumable_path=None, simple_multipart=True, simple_path='/upload/v1beta1/{+parent}/ragFiles:upload')}

    def Upload(self, request, global_params=None, upload=None):
        """Upload a file into a RagCorpus.

      Args:
        request: (AiplatformMediaUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (GoogleCloudAiplatformV1beta1UploadRagFileResponse) The response message.
      """
        config = self.GetMethodConfig('Upload')
        upload_config = self.GetUploadConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}/ragFiles:upload', http_method='POST', method_id='aiplatform.media.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/ragFiles:upload', request_field='googleCloudAiplatformV1beta1UploadRagFileRequest', request_type_name='AiplatformMediaUploadRequest', response_type_name='GoogleCloudAiplatformV1beta1UploadRagFileResponse', supports_download=False)