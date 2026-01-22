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