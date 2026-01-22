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
def _ranged_download(self, bucket, key, filename, object_size, extra_args, callback):
    downloader = MultipartDownloader(self._client, self._config, self._osutil)
    downloader.download_file(bucket, key, filename, object_size, extra_args, callback)