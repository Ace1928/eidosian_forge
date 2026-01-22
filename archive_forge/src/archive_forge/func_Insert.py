import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def Insert(self, request, global_params=None, upload=None):
    """Stores a new object and metadata.

      Args:
        request: (StorageObjectsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (Object) The response message.
      """
    config = self.GetMethodConfig('Insert')
    upload_config = self.GetUploadConfig('Insert')
    return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)