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
def PerformDownload(start_byte, progress_callback):
    """Downloads the source object in chunks.

      This function checks the stop_download event and exits early if it is set.
      It should be set when there is an error during the daisy-chain upload,
      then this function can be called again with the upload's current position
      as start_byte.

      Args:
        start_byte: Byte from which to begin the download.
        progress_callback: Optional callback function for progress
            notifications. Receives calls with arguments
            (bytes_transferred, total_size).
      """
    self.download_started.set()
    try:
        while start_byte + self._download_chunk_size < self.src_obj_size:
            self.gsutil_api.GetObjectMedia(self.src_url.bucket_name, self.src_url.object_name, BufferWrapper(self), compressed_encoding=self.compressed_encoding, start_byte=start_byte, end_byte=start_byte + self._download_chunk_size - 1, generation=self.src_url.generation, object_size=self.src_obj_size, download_strategy=CloudApi.DownloadStrategy.ONE_SHOT, provider=self.src_url.scheme, progress_callback=progress_callback, decryption_tuple=self.decryption_tuple)
            if self.stop_download.is_set():
                self.stop_download.clear()
                return
            start_byte += self._download_chunk_size
        self.gsutil_api.GetObjectMedia(self.src_url.bucket_name, self.src_url.object_name, BufferWrapper(self), compressed_encoding=self.compressed_encoding, start_byte=start_byte, generation=self.src_url.generation, object_size=self.src_obj_size, download_strategy=CloudApi.DownloadStrategy.ONE_SHOT, provider=self.src_url.scheme, progress_callback=progress_callback, decryption_tuple=self.decryption_tuple)
    except Exception as e:
        with self.download_exception_lock:
            self.download_exception = e
            raise