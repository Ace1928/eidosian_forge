from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import os
import pkgutil
import six
import gslib.cloud_api
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
class MockDownloadCloudApi(gslib.cloud_api.CloudApi):
    """Mock CloudApi that implements GetObjectMedia for testing."""

    def __init__(self, write_values):
        """Initialize the mock that will be used by the download thread.

      Args:
        write_values: List of values that will be used for calls to write(),
            in order, by the download thread. An Exception class may be part of
            the list; if so, the Exception will be raised after previous
            values are consumed.
      """
        self._write_values = write_values
        self.get_calls = 0

    def GetObjectMedia(self, unused_bucket_name, unused_object_name, download_stream, start_byte=0, end_byte=None, **kwargs):
        """Writes self._write_values to the download_stream."""
        self.get_calls += 1
        bytes_read = 0
        for write_value in self._write_values:
            if bytes_read < start_byte:
                bytes_read += len(write_value)
                continue
            if end_byte and bytes_read >= end_byte:
                break
            if isinstance(write_value, Exception):
                raise write_value
            download_stream.write(write_value)
            bytes_read += len(write_value)