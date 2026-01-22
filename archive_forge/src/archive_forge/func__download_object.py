from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
import boto3
import botocore
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import headers_util
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.api_lib.storage import xml_metadata_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import s3transfer
def _download_object(self, cloud_resource, download_stream, digesters, progress_callback, start_byte, end_byte):
    get_object_args = {'Bucket': cloud_resource.bucket, 'Key': cloud_resource.name}
    if end_byte is None:
        get_object_args['Range'] = 'bytes={}-'.format(start_byte)
    else:
        get_object_args['Range'] = 'bytes={}-{}'.format(start_byte, end_byte)
    if cloud_resource.generation is not None:
        get_object_args['VersionId'] = str(cloud_resource.generation)
    response = self.client.get_object(**get_object_args)
    processed_bytes = start_byte
    for chunk in response['Body'].iter_chunks(scaled_integer.ParseInteger(properties.VALUES.storage.download_chunk_size.Get())):
        download_stream.write(chunk)
        for hash_object in digesters.values():
            hash_object.update(chunk)
        processed_bytes += len(chunk)
        if progress_callback:
            progress_callback(processed_bytes)
    return response.get('ContentEncoding')