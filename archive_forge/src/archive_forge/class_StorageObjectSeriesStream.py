from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.dataproc import exceptions as dp_exceptions
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six.moves.urllib.parse
class StorageObjectSeriesStream(object):
    """I/O Stream-like class for communicating via a sequence of GCS objects."""

    def __init__(self, path, storage_client=None):
        """Construct a StorageObjectSeriesStream for a specific gcs path.

    Args:
      path: A GCS object prefix which will be the base of the objects used to
          communicate across the channel.
      storage_client: a StorageClient for accessing GCS.

    Returns:
      The constructed stream.
    """
        self._base_path = path
        self._gcs = storage_client or StorageClient()
        self._open = True
        self._current_object_index = 0
        self._current_object_pos = 0

    @property
    def open(self):
        """Whether the stream is open."""
        return self._open

    def Close(self):
        """Close the stream."""
        self._open = False

    def _AssertOpen(self):
        if not self.open:
            raise ValueError('I/O operation on closed stream.')

    def _GetObject(self, i):
        """Get the ith object in the series."""
        path = '{0}.{1:09d}'.format(self._base_path, i)
        return self._gcs.GetObject(GetObjectRef(path, self._gcs.messages))

    def ReadIntoWritable(self, writable, n=sys.maxsize):
        """Read from this stream into a writable.

    Reads at most n bytes, or until it sees there is not a next object in the
    series. This will block for the duration of each object's download,
    and possibly indefinitely if new objects are being added to the channel
    frequently enough.

    Args:
      writable: The stream-like object that implements write(<string>) to
          write into.
      n: A maximum number of bytes to read. Defaults to sys.maxsize
        (usually ~4 GB).

    Raises:
      ValueError: If the stream is closed or objects in the series are
        detected to shrink.

    Returns:
      The number of bytes read.
    """
        self._AssertOpen()
        bytes_read = 0
        object_info = None
        max_bytes_to_read = n
        while bytes_read < max_bytes_to_read:
            next_object_info = self._GetObject(self._current_object_index + 1)
            if not object_info or next_object_info:
                try:
                    object_info = self._GetObject(self._current_object_index)
                except apitools_exceptions.HttpError as error:
                    log.warning('Failed to fetch GCS output:\n%s', error)
                    break
                if not object_info:
                    break
            new_bytes_available = object_info.size - self._current_object_pos
            if new_bytes_available < 0:
                raise ValueError('Object [{0}] shrunk.'.format(object_info.name))
            if object_info.size == 0:
                self.Close()
                break
            bytes_left_to_read = max_bytes_to_read - bytes_read
            new_bytes_to_read = min(bytes_left_to_read, new_bytes_available)
            if new_bytes_to_read > 0:
                download = self._gcs.BuildObjectStream(writable, object_info)
                download.GetRange(self._current_object_pos, self._current_object_pos + new_bytes_to_read - 1)
                self._current_object_pos += new_bytes_to_read
                bytes_read += new_bytes_to_read
            object_finished = next_object_info and self._current_object_pos == object_info.size
            if object_finished:
                object_info = next_object_info
                self._current_object_index += 1
                self._current_object_pos = 0
                continue
            else:
                break
        return bytes_read