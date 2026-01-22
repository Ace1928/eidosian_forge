from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import json
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer as apitools_transfer
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage import gcs_iam_util
from googlecloudsdk.api_lib.storage import headers_util
from googlecloudsdk.api_lib.storage.gcs_json import download
from googlecloudsdk.api_lib.storage.gcs_json import error_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.storage.gcs_json import patch_apitools_messages
from googlecloudsdk.api_lib.storage.gcs_json import upload
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
from six.moves import urllib
class _StorageStreamResponseHandler(requests.ResponseHandler):
    """Handler for writing the streaming response to the download stream."""

    def __init__(self):
        """Initializes response handler for requests downloads."""
        super(_StorageStreamResponseHandler, self).__init__(use_stream=True)
        self._stream = None
        self._digesters = {}
        self._processed_bytes = (0,)
        self._progress_callback = None
        self._size = None
        self._chunk_size = scaled_integer.ParseInteger(properties.VALUES.storage.download_chunk_size.Get())
        self._progress_callback_threshold = max(MINIMUM_PROGRESS_CALLBACK_THRESHOLD, self._chunk_size)

    def update_destination_info(self, stream, size, digesters=None, download_strategy=None, processed_bytes=0, progress_callback=None):
        """Updates the stream handler with destination information.

    The download_http_client object is stored on the gcs_api object. This allows
    resusing the same http_client when the gcs_api is cached using
    threading.local, which improves performance.
    Since this same object gets used for multiple downloads, we need to update
    the stream handler with the current active download's destination.

    Args:
      stream (stream): Local stream to write downloaded data to.
      size (int): The amount of data in bytes to be downloaded.
      digesters (dict<HashAlgorithm, hashlib object> | None): For updating hash
        digests of downloaded objects on the fly.
      download_strategy (DownloadStrategy): Indicates if user wants to retry
        download on failures.
      processed_bytes (int): For keeping track of how much progress has been
        made.
      progress_callback (func<int>): Accepts processed_bytes and submits
        progress info for aggregation.
    """
        self._stream = stream
        self._size = size
        self._digesters = digesters if digesters is not None else {}
        self._download_strategy = download_strategy
        self._processed_bytes = processed_bytes
        self._progress_callback = progress_callback
        self._start_byte = self._processed_bytes

    def handle(self, source_stream):
        if self._stream is None:
            raise command_errors.Error('Stream was not found.')
        destination_pipe_is_broken = False
        bytes_since_last_progress_callback = 0
        while True:
            data = source_stream.read(self._chunk_size)
            if data:
                try:
                    self._stream.write(data)
                except OSError as e:
                    if e.errno == errno.EPIPE and self._download_strategy is cloud_api.DownloadStrategy.ONE_SHOT:
                        log.info('Writing to download stream raised broken pipe error.')
                        destination_pipe_is_broken = True
                        break
                    raise
                for hash_object in self._digesters.values():
                    hash_object.update(data)
                self._processed_bytes += len(data)
                bytes_since_last_progress_callback += len(data)
                if self._progress_callback and bytes_since_last_progress_callback >= self._progress_callback_threshold:
                    self._progress_callback(self._processed_bytes)
                    bytes_since_last_progress_callback = bytes_since_last_progress_callback - self._progress_callback_threshold
            else:
                if self._progress_callback and bytes_since_last_progress_callback:
                    self._progress_callback(self._processed_bytes)
                break
        total_downloaded_data = self._processed_bytes - self._start_byte
        if self._size != total_downloaded_data and (not destination_pipe_is_broken):
            message = 'Download not completed. Target size={}, downloaded data={}'.format(self._size, total_downloaded_data)
            log.debug(message)
            raise cloud_errors.RetryableApiError(message)