from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import os
import threading
import time
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import CloudApi
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
class BufferWrapper(object):
    """Wraps the download file pointer to use our in-memory buffer."""

    def __init__(self, daisy_chain_wrapper, mode='b'):
        """Provides a buffered write interface for a file download.

    Args:
      daisy_chain_wrapper: DaisyChainWrapper instance to use for buffer and
                           locking.
    """
        self.daisy_chain_wrapper = daisy_chain_wrapper
        if hasattr(daisy_chain_wrapper, 'mode'):
            self.mode = daisy_chain_wrapper.mode
        else:
            self.mode = mode

    def write(self, data):
        """Waits for space in the buffer, then writes data to the buffer."""
        while True:
            with self.daisy_chain_wrapper.lock:
                if self.daisy_chain_wrapper.bytes_buffered < self.daisy_chain_wrapper.max_buffer_size:
                    break
            time.sleep(0)
        data_len = len(data)
        if data_len:
            with self.daisy_chain_wrapper.lock:
                self.daisy_chain_wrapper.buffer.append(data)
                self.daisy_chain_wrapper.bytes_buffered += data_len