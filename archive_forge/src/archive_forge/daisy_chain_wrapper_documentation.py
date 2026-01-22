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
Downloads the source object in chunks.

      This function checks the stop_download event and exits early if it is set.
      It should be set when there is an error during the daisy-chain upload,
      then this function can be called again with the upload's current position
      as start_byte.

      Args:
        start_byte: Byte from which to begin the download.
        progress_callback: Optional callback function for progress
            notifications. Receives calls with arguments
            (bytes_transferred, total_size).
      