from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import functools
import os
from googlecloudsdk.api_lib.storage import retry_util as storage_retry_util
from googlecloudsdk.api_lib.storage.gcs_grpc import grpc_util
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import retry_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
import six
class StreamingUpload(RecoverableUpload):
    """Uploads objects from a stream with support for error recovery in-flight.

    Stream is split into chunks of size set by property upload_chunk_size,
    rounded down to be a multiple of MAX_WRITE_CHUNK_BYTES.

    For example if upload_chunk_size is 7 MiB and MAX_WRITE_CHUNK_BYTES is
    2 MiB, it will be rounded down to 6 MiB. If upload_chunk_size is less than
    MAX_WRITE_CHUNK_BYTES, it will be equal to MAX_WRITE_CHUNK_BYTES.
  """

    def __init__(self, client, source_stream, destination_resource, request_config, source_resource=None):
        super(StreamingUpload, self).__init__(client, source_stream, destination_resource, request_config, source_resource)
        self._log_chunk_warning = False
        self._chunk_size = self._get_chunk_size()

    def _get_chunk_size(self):
        """Returns the chunk size corrected to be multiple of MAX_WRITE_CHUNK_BYTES.

    It also sets the attribute _should_log_message if it is needed to log
    the warning message.

    Look at the docstring on StreamingUpload class.

    Returns:
      (int) The chunksize value corrected.
    """
        initial_chunk_size = scaled_integer.ParseInteger(properties.VALUES.storage.upload_chunk_size.Get())
        if initial_chunk_size >= self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES:
            adjust_chunk_size = initial_chunk_size % self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES
            if adjust_chunk_size > 0:
                self._log_chunk_warning = True
            return initial_chunk_size - adjust_chunk_size
        self._log_chunk_warning = True
        return self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES

    def _log_message(self):
        if not self._log_chunk_warning:
            return
        initial_chunk_size = scaled_integer.ParseInteger(properties.VALUES.storage.upload_chunk_size.Get())
        log.warning('Data will be sent in chunks of {} instead of {}, as configured in the storage/upload_chunk_size config value.'.format(scaled_integer.FormatBinaryNumber(self._chunk_size), scaled_integer.FormatBinaryNumber(initial_chunk_size)))

    def _perform_upload(self, upload_id):
        self._log_message()
        response = None
        while True:
            response = self._call_write_object(upload_id)
            if self._source_stream_finished:
                break
            self._start_offset = response.persisted_size
        return response