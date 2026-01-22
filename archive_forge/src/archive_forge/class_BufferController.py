from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import upload_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class BufferController:
    """Manages a  bidirectional buffer to read and write simultaneously.

  Attributes:
    buffer_queue (collections.deque): The underlying queue that acts like a
      buffer for the streams
    buffer_condition (threading.Condition): The condition object used for
      waiting based on the underlying buffer_queue state.
      All threads waiting on this condition are notified when data is added or
      removed from buffer_queue. Streams that write to the buffer wait on this
      condition until the buffer has space, and streams that read from the
      buffer wait on this condition until the buffer has data.
    shutdown_event (threading.Event): Used for signaling the operations to
      terminate.
    writable_stream (_WritableStream): Stream that writes to the buffer.
    readable_stream (_ReadableStream): Stream that reads from the buffer.
    exception_raised (Exception): Stores the Exception instance responsible for
      termination of the operation.
  """

    def __init__(self, source_resource, destination_scheme, user_request_args=None, progress_callback=None):
        """Initializes BufferController.

    Args:
      source_resource (resource_reference.ObjectResource): Must
        contain the full object path of existing object.
      destination_scheme (storage_url.ProviderPrefix): The destination provider.
      user_request_args (UserRequestArgs|None): Values for RequestConfig.
      progress_callback (progress_callbacks.FilesAndBytesProgressCallback):
        Accepts processed bytes and submits progress info for aggregation.
    """
        self._source_resource = source_resource
        self._user_request_args = user_request_args
        self.buffer_queue = collections.deque()
        self.buffer_condition = threading.Condition()
        self.shutdown_event = threading.Event()
        self.writable_stream = _WritableStream(self.buffer_queue, self.buffer_condition, self.shutdown_event)
        destination_capabilities = api_factory.get_capabilities(destination_scheme)
        self.readable_stream = _ReadableStream(self.buffer_queue, self.buffer_condition, self.shutdown_event, self._source_resource.size, restart_download_callback=self.restart_download, progress_callback=progress_callback, seekable=cloud_api.Capability.DAISY_CHAIN_SEEKABLE_UPLOAD_STREAM in destination_capabilities)
        self._download_thread = None
        self.exception_raised = None

    def _run_download(self, start_byte):
        """Performs the download operation."""
        request_config = request_config_factory.get_request_config(self._source_resource.storage_url, user_request_args=self._user_request_args)
        client = api_factory.get_api(self._source_resource.storage_url.scheme)
        try:
            if self._source_resource.size != 0:
                client.download_object(self._source_resource, self.writable_stream, request_config, start_byte=start_byte, download_strategy=cloud_api.DownloadStrategy.ONE_SHOT)
        except _AbruptShutdownError:
            pass
        except Exception as e:
            self.shutdown(e)

    def start_download_thread(self, start_byte=0):
        self._download_thread = threading.Thread(target=self._run_download, args=(start_byte,))
        self._download_thread.start()

    def wait_for_download_thread_to_terminate(self):
        if self._download_thread is not None:
            self._download_thread.join()

    def restart_download(self, start_byte):
        """Restarts the download_thread.

    Args:
      start_byte (int): The start byte for the new download call.
    """
        self.shutdown_event.set()
        with self.buffer_condition:
            self.buffer_condition.notify_all()
        self.wait_for_download_thread_to_terminate()
        self.buffer_queue.clear()
        self.shutdown_event.clear()
        self.start_download_thread(start_byte)

    def shutdown(self, error):
        """Sets the shutdown event and stores the error to re-raise later.

    Args:
      error (Exception): The error responsible for triggering shutdown.
    """
        self.shutdown_event.set()
        with self.buffer_condition:
            self.buffer_condition.notify_all()
            self.exception_raised = error