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
def _upload_one_part(self, filename, bucket, key, upload_id, part_size, extra_args, callback, part_number):
    open_chunk_reader = self._os.open_file_chunk_reader
    with open_chunk_reader(filename, part_size * (part_number - 1), part_size, callback) as body:
        response = self._client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=part_number, Body=body, **extra_args)
        etag = response['ETag']
        return {'ETag': etag, 'PartNumber': part_number}