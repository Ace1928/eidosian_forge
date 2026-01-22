from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _CloudStorageConfig(self, bucket, file_prefix, file_suffix, file_datetime_format, max_bytes, max_duration, output_format, write_metadata):
    """Builds CloudStorageConfig message from argument values.

    Args:
      bucket (str): The name for the Cloud Storage bucket.
      file_prefix (str): The prefix for Cloud Storage filename.
      file_suffix (str): The suffix for Cloud Storage filename.
      file_datetime_format (str): The custom datetime format string for Cloud
        Storage filename.
      max_bytes (int): The maximum bytes that can be written to a Cloud Storage
        file before a new file is created.
      max_duration (str): The maximum duration that can elapse before a new
        Cloud Storage file is created.
      output_format (str): The output format for data written to Cloud Storage.
      write_metadata (bool): Whether or not to write the subscription name and
        other metadata in the output.

    Returns:
      CloudStorageConfig message or None
    """
    if bucket:
        cloud_storage_config = self.messages.CloudStorageConfig(bucket=bucket, filenamePrefix=file_prefix, filenameSuffix=file_suffix, filenameDatetimeFormat=file_datetime_format, maxBytes=max_bytes, maxDuration=max_duration)
        if output_format == 'text':
            cloud_storage_config.textConfig = self.messages.TextConfig()
        elif output_format == 'avro':
            cloud_storage_config.avroConfig = self.messages.AvroConfig(writeMetadata=write_metadata if write_metadata else False)
        return cloud_storage_config
    return None