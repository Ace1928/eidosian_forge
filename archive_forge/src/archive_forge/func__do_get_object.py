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
def _do_get_object(self, bucket, key, filename, extra_args, callback):
    response = self._client.get_object(Bucket=bucket, Key=key, **extra_args)
    streaming_body = StreamReaderProgress(response['Body'], callback)
    with self._osutil.open(filename, 'wb') as f:
        for chunk in iter(lambda: streaming_body.read(8192), b''):
            f.write(chunk)