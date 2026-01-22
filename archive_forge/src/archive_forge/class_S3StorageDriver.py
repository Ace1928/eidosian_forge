import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
class S3StorageDriver(AWSDriver, BaseS3StorageDriver):
    name = 'Amazon S3'
    connectionCls = S3SignatureV4Connection
    region_name = 'us-east-1'

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region=None, token=None, **kwargs):
        if hasattr(self, 'region_name') and (not region):
            region = self.region_name
        self.region_name = region
        if region and region not in REGION_TO_HOST_MAP.keys():
            raise ValueError('Invalid or unsupported region: %s' % region)
        self.name = 'Amazon S3 (%s)' % region
        if host is None:
            host = REGION_TO_HOST_MAP[region]
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, token=token, **kwargs)

    @classmethod
    def list_regions(self):
        return REGION_TO_HOST_MAP.keys()

    def get_object_cdn_url(self, obj, ex_expiry=S3_CDN_URL_EXPIRY_HOURS):
        """
        Return a "presigned URL" for read-only access to object

        AWS only - requires AWS signature V4 authentication.

        :param obj: Object instance.
        :type  obj: :class:`Object`

        :param ex_expiry: The number of hours after which the URL expires.
                          Defaults to 24 hours or the value of the environment
                          variable "LIBCLOUD_S3_STORAGE_CDN_URL_EXPIRY_HOURS",
                          if set.
        :type  ex_expiry: ``float``

        :return: Presigned URL for the object.
        :rtype: ``str``
        """
        object_path = self._get_object_path(obj.container, obj.name)
        now = datetime.utcnow()
        duration_seconds = int(ex_expiry * 3600)
        credparts = (self.key, now.strftime(S3_CDN_URL_DATE_FORMAT), self.region, 's3', 'aws4_request')
        params_to_sign = {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': '/'.join(credparts), 'X-Amz-Date': now.strftime(S3_CDN_URL_DATETIME_FORMAT), 'X-Amz-Expires': duration_seconds, 'X-Amz-SignedHeaders': 'host'}
        headers_to_sign = {'host': self.connection.host}
        signature = self.connection.signer._get_signature(params=params_to_sign, headers=headers_to_sign, dt=now, method='GET', path=object_path, data=UnsignedPayloadSentinel)
        params = params_to_sign.copy()
        params['X-Amz-Signature'] = signature
        return '{scheme}://{host}:{port}{path}?{params}'.format(scheme='https' if self.secure else 'http', host=self.connection.host, port=self.connection.port, path=object_path, params=urlencode(params))