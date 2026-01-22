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
class _ReadableStream:
    """A read-only stream that reads from the buffer queue."""

    def __init__(self, buffer_queue, buffer_condition, shutdown_event, end_position, restart_download_callback, progress_callback=None, seekable=True):
        """Initializes ReadableStream.

    Args:
      buffer_queue (collections.deque): The underlying queue from which the data
        gets read.
      buffer_condition (threading.Condition): The condition object to wait on if
        the buffer is empty.
      shutdown_event (threading.Event): Used for signaling the thread to
        terminate.
      end_position (int): Position at which the stream reading stops. This is
        usually the total size of the data that gets read.
      restart_download_callback (func): This must be the
        BufferController.restart_download function.
      progress_callback (progress_callbacks.FilesAndBytesProgressCallback):
        Accepts processed bytes and submits progress info for aggregation.
      seekable (bool): Value for the "seekable" method call.
    """
        self._buffer_queue = buffer_queue
        self._buffer_condition = buffer_condition
        self._end_position = end_position
        self._shutdown_event = shutdown_event
        self._position = 0
        self._unused_data_from_previous_read = b''
        self._progress_callback = progress_callback
        self._restart_download_callback = restart_download_callback
        self._bytes_read_since_last_progress_callback = 0
        self._seekable = seekable
        self._is_closed = False

    def _restart_download(self, offset):
        self._restart_download_callback(offset)
        self._unused_data_from_previous_read = b''
        self._bytes_read_since_last_progress_callback = 0
        self._position = offset

    def read(self, size=-1):
        """Reads size bytes from the buffer queue and returns it.

    This method will be blocked if the buffer_queue is empty.
    If size > length of data available, the entire data is sent over.

    Args:
      size (int): The number of bytes to be read.

    Returns:
      Bytes of length 'size'. May return bytes of length less than the size
        if there are no more bytes left to be read.

    Raises:
      _AbruptShutdownError: If self._shudown_event was set.
      storage.errors.Error: If size is not within the allowed range of
        [-1, MAX_ALLOWED_READ_SIZE] OR
        If size is -1 but the object size is greater than MAX_ALLOWED_READ_SIZE.
    """
        if size == 0:
            return b''
        if size > _MAX_ALLOWED_READ_SIZE:
            raise errors.Error('Invalid HTTP read size {} during daisy chain operation, expected -1 <= size <= {} bytes.'.format(size, _MAX_ALLOWED_READ_SIZE))
        if size == -1:
            if self._end_position <= _MAX_ALLOWED_READ_SIZE:
                chunk_size = self._end_position
            else:
                raise errors.Error('Read with size=-1 is not allowed for object size > {} bytes to prevent reading large objects in-memory.'.format(_MAX_ALLOWED_READ_SIZE))
        else:
            chunk_size = size
        result = io.BytesIO()
        bytes_read = 0
        while bytes_read < chunk_size and self._position < self._end_position:
            if not self._unused_data_from_previous_read:
                with self._buffer_condition:
                    while not self._buffer_queue and (not self._shutdown_event.is_set()):
                        self._buffer_condition.wait()
                    if self._shutdown_event.is_set():
                        raise _AbruptShutdownError()
                    data = self._buffer_queue.popleft()
                    self._buffer_condition.notify_all()
            else:
                if self._shutdown_event.is_set():
                    raise _AbruptShutdownError()
                data = self._unused_data_from_previous_read
            if bytes_read + len(data) > chunk_size:
                self._unused_data_from_previous_read = data[chunk_size - bytes_read:]
                data_to_return = data[:chunk_size - bytes_read]
            else:
                self._unused_data_from_previous_read = b''
                data_to_return = data
            result.write(data_to_return)
            bytes_read += len(data_to_return)
            self._position += len(data_to_return)
        result_data = result.getvalue()
        if result_data and self._progress_callback:
            self._bytes_read_since_last_progress_callback += len(result_data)
            if self._bytes_read_since_last_progress_callback >= _PROGRESS_CALLBACK_THRESHOLD:
                self._bytes_read_since_last_progress_callback = 0
                self._progress_callback(self._position)
        return result_data

    def seek(self, offset, whence=os.SEEK_SET):
        """Seek to the given offset position.

    Ideally, seek changes the stream position to the given byte offset.
    But we only handle resumable retry for S3 to GCS transfers at this time,
    which means, seek will be called only by the Apitools library.
    Since Apitools calls seek only for limited cases, we avoid implementing
    seek for all possible cases here in order to avoid unnecessary complexity
    in the code.

    Following are the cases where Apitools calls seek:
    1) At the end of the transfer
    https://github.com/google/apitools/blob/ca2094556531d61e741dc2954fdfccbc650cdc32/apitools/base/py/transfer.py#L986
    to determine if it has read everything from the stream.
    2) For any transient errors during uploads to seek back to a particular
    position. This call is always made with whence == os.SEEK_SET.

    Args:
      offset (int): Defines the position realative to the `whence` where the
        current position of the stream should be moved.
      whence (int): The reference relative to which offset is interpreted.
        Values for whence are: os.SEEK_SET or 0 - start of the stream
        (thedefault). os.SEEK_END or 2 - end of the stream. We do not support
        other os.SEEK_* constants.

    Returns:
      (int) The current position.

    Raises:
      Error:
        If seek is called with whence == os.SEEK_END for offset not
        equal to the last position.
        If seek is called with whence == os.SEEK_CUR.
    """
        if whence == os.SEEK_END:
            if offset:
                raise errors.Error('Non-zero offset from os.SEEK_END is not allowed.Offset: {}.'.format(offset))
        elif whence == os.SEEK_SET:
            if offset != self._position:
                self._restart_download(offset)
        else:
            raise errors.Error('Seek is only supported for os.SEEK_END and os.SEEK_SET.')
        return self._position

    def seekable(self):
        """Returns True if the stream should be treated as a seekable stream."""
        return self._seekable

    def tell(self):
        """Returns the current position."""
        return self._position

    def close(self):
        """Updates progress callback if needed."""
        if self._is_closed:
            return
        if self._progress_callback and (self._bytes_read_since_last_progress_callback or self._end_position == 0):
            self._bytes_read_since_last_progress_callback = 0
            self._progress_callback(self._position)
        self._is_closed = True