import base64
import functools
import hashlib
import logging
import mimetypes
import os
import time
from collections import defaultdict
from contextlib import suppress
from ftplib import FTP
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import DefaultDict, Optional, Set, Union
from urllib.parse import urlparse
from itemadapter import ItemAdapter
from twisted.internet import defer, threads
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.media import MediaPipeline
from scrapy.settings import Settings
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.datatypes import CaseInsensitiveDict
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import md5sum
from scrapy.utils.python import to_bytes
from scrapy.utils.request import referer_str
class S3FilesStore:
    AWS_ACCESS_KEY_ID = None
    AWS_SECRET_ACCESS_KEY = None
    AWS_SESSION_TOKEN = None
    AWS_ENDPOINT_URL = None
    AWS_REGION_NAME = None
    AWS_USE_SSL = None
    AWS_VERIFY = None
    POLICY = 'private'
    HEADERS = {'Cache-Control': 'max-age=172800'}

    def __init__(self, uri):
        if not is_botocore_available():
            raise NotConfigured('missing botocore library')
        import botocore.session
        session = botocore.session.get_session()
        self.s3_client = session.create_client('s3', aws_access_key_id=self.AWS_ACCESS_KEY_ID, aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY, aws_session_token=self.AWS_SESSION_TOKEN, endpoint_url=self.AWS_ENDPOINT_URL, region_name=self.AWS_REGION_NAME, use_ssl=self.AWS_USE_SSL, verify=self.AWS_VERIFY)
        if not uri.startswith('s3://'):
            raise ValueError(f"Incorrect URI scheme in {uri}, expected 's3'")
        self.bucket, self.prefix = uri[5:].split('/', 1)

    def stat_file(self, path, info):

        def _onsuccess(boto_key):
            checksum = boto_key['ETag'].strip('"')
            last_modified = boto_key['LastModified']
            modified_stamp = time.mktime(last_modified.timetuple())
            return {'checksum': checksum, 'last_modified': modified_stamp}
        return self._get_boto_key(path).addCallback(_onsuccess)

    def _get_boto_key(self, path):
        key_name = f'{self.prefix}{path}'
        return threads.deferToThread(self.s3_client.head_object, Bucket=self.bucket, Key=key_name)

    def persist_file(self, path, buf, info, meta=None, headers=None):
        """Upload file to S3 storage"""
        key_name = f'{self.prefix}{path}'
        buf.seek(0)
        extra = self._headers_to_botocore_kwargs(self.HEADERS)
        if headers:
            extra.update(self._headers_to_botocore_kwargs(headers))
        return threads.deferToThread(self.s3_client.put_object, Bucket=self.bucket, Key=key_name, Body=buf, Metadata={k: str(v) for k, v in (meta or {}).items()}, ACL=self.POLICY, **extra)

    def _headers_to_botocore_kwargs(self, headers):
        """Convert headers to botocore keyword arguments."""
        mapping = CaseInsensitiveDict({'Content-Type': 'ContentType', 'Cache-Control': 'CacheControl', 'Content-Disposition': 'ContentDisposition', 'Content-Encoding': 'ContentEncoding', 'Content-Language': 'ContentLanguage', 'Content-Length': 'ContentLength', 'Content-MD5': 'ContentMD5', 'Expires': 'Expires', 'X-Amz-Grant-Full-Control': 'GrantFullControl', 'X-Amz-Grant-Read': 'GrantRead', 'X-Amz-Grant-Read-ACP': 'GrantReadACP', 'X-Amz-Grant-Write-ACP': 'GrantWriteACP', 'X-Amz-Object-Lock-Legal-Hold': 'ObjectLockLegalHoldStatus', 'X-Amz-Object-Lock-Mode': 'ObjectLockMode', 'X-Amz-Object-Lock-Retain-Until-Date': 'ObjectLockRetainUntilDate', 'X-Amz-Request-Payer': 'RequestPayer', 'X-Amz-Server-Side-Encryption': 'ServerSideEncryption', 'X-Amz-Server-Side-Encryption-Aws-Kms-Key-Id': 'SSEKMSKeyId', 'X-Amz-Server-Side-Encryption-Context': 'SSEKMSEncryptionContext', 'X-Amz-Server-Side-Encryption-Customer-Algorithm': 'SSECustomerAlgorithm', 'X-Amz-Server-Side-Encryption-Customer-Key': 'SSECustomerKey', 'X-Amz-Server-Side-Encryption-Customer-Key-Md5': 'SSECustomerKeyMD5', 'X-Amz-Storage-Class': 'StorageClass', 'X-Amz-Tagging': 'Tagging', 'X-Amz-Website-Redirect-Location': 'WebsiteRedirectLocation'})
        extra = {}
        for key, value in headers.items():
            try:
                kwarg = mapping[key]
            except KeyError:
                raise TypeError(f'Header "{key}" is not supported by botocore')
            else:
                extra[kwarg] = value
        return extra