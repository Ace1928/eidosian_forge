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
class MultipartUploader(object):
    UPLOAD_PART_ARGS = ['SSECustomerKey', 'SSECustomerAlgorithm', 'SSECustomerKeyMD5', 'RequestPayer']

    def __init__(self, client, config, osutil, executor_cls=concurrent.futures.ThreadPoolExecutor):
        self._client = client
        self._config = config
        self._os = osutil
        self._executor_cls = executor_cls

    def _extra_upload_part_args(self, extra_args):
        upload_parts_args = {}
        for key, value in extra_args.items():
            if key in self.UPLOAD_PART_ARGS:
                upload_parts_args[key] = value
        return upload_parts_args

    def upload_file(self, filename, bucket, key, callback, extra_args):
        response = self._client.create_multipart_upload(Bucket=bucket, Key=key, **extra_args)
        upload_id = response['UploadId']
        try:
            parts = self._upload_parts(upload_id, filename, bucket, key, callback, extra_args)
        except Exception as e:
            logger.debug('Exception raised while uploading parts, aborting multipart upload.', exc_info=True)
            self._client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            raise S3UploadFailedError('Failed to upload %s to %s: %s' % (filename, '/'.join([bucket, key]), e))
        self._client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={'Parts': parts})

    def _upload_parts(self, upload_id, filename, bucket, key, callback, extra_args):
        upload_parts_extra_args = self._extra_upload_part_args(extra_args)
        parts = []
        part_size = self._config.multipart_chunksize
        num_parts = int(math.ceil(self._os.get_file_size(filename) / float(part_size)))
        max_workers = self._config.max_concurrency
        with self._executor_cls(max_workers=max_workers) as executor:
            upload_partial = functools.partial(self._upload_one_part, filename, bucket, key, upload_id, part_size, upload_parts_extra_args, callback)
            for part in executor.map(upload_partial, range(1, num_parts + 1)):
                parts.append(part)
        return parts

    def _upload_one_part(self, filename, bucket, key, upload_id, part_size, extra_args, callback, part_number):
        open_chunk_reader = self._os.open_file_chunk_reader
        with open_chunk_reader(filename, part_size * (part_number - 1), part_size, callback) as body:
            response = self._client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=part_number, Body=body, **extra_args)
            etag = response['ETag']
            return {'ETag': etag, 'PartNumber': part_number}