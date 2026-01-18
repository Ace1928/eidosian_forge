from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage.gcs_grpc import download
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import upload
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
def upload_object(self, source_stream, destination_resource, request_config, posix_to_set=None, serialization_data=None, source_resource=None, tracker_callback=None, upload_strategy=cloud_api.UploadStrategy.SIMPLE):
    """See super class."""
    client = self._get_gapic_client()
    source_path = self._get_source_path(source_resource)
    should_gzip_in_flight = gzip_util.should_gzip_in_flight(request_config.gzip_settings, source_path)
    if should_gzip_in_flight:
        raise core_exceptions.InternalError('Gzip transport encoding is not supported with GRPC API, please use the JSON API instead, changing the storage/preferred_api config value to json.')
    if upload_strategy == cloud_api.UploadStrategy.SIMPLE:
        uploader = upload.SimpleUpload(client=client, source_stream=source_stream, destination_resource=destination_resource, request_config=request_config, source_resource=source_resource)
    elif upload_strategy == cloud_api.UploadStrategy.RESUMABLE:
        uploader = upload.ResumableUpload(client=client, source_stream=source_stream, destination_resource=destination_resource, request_config=request_config, serialization_data=serialization_data, source_resource=source_resource, tracker_callback=tracker_callback)
    else:
        uploader = upload.StreamingUpload(client=client, source_stream=source_stream, destination_resource=destination_resource, request_config=request_config, source_resource=source_resource)
    response = uploader.run()
    return metadata_util.get_object_resource_from_grpc_object(response.resource)