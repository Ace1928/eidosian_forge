import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
def _download_range(self, bucket, key, filename, part_size, num_parts, callback, part_index):
    try:
        range_param = self._calculate_range_param(part_size, part_index, num_parts)
        max_attempts = self._config.num_download_attempts
        last_exception = None
        for i in range(max_attempts):
            try:
                logger.debug('Making get_object call.')
                response = self._client.get_object(Bucket=bucket, Key=key, Range=range_param)
                streaming_body = StreamReaderProgress(response['Body'], callback)
                buffer_size = 1024 * 16
                current_index = part_size * part_index
                for chunk in iter(lambda: streaming_body.read(buffer_size), b''):
                    self._ioqueue.put((current_index, chunk))
                    current_index += len(chunk)
                return
            except (socket.timeout, socket.error, ReadTimeoutError, IncompleteReadError) as e:
                logger.debug('Retrying exception caught (%s), retrying request, (attempt %s / %s)', e, i, max_attempts, exc_info=True)
                last_exception = e
                continue
        raise RetriesExceededError(last_exception)
    finally:
        logger.debug('EXITING _download_range for part: %s', part_index)