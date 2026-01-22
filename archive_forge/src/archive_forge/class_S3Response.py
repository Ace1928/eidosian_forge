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
class S3Response(AWSBaseResponse):
    namespace = None
    valid_response_codes = [httplib.NOT_FOUND, httplib.CONFLICT, httplib.BAD_REQUEST, httplib.PARTIAL_CONTENT]

    def success(self):
        i = int(self.status)
        return 200 <= i <= 299 or i in self.valid_response_codes

    def parse_error(self):
        if self.status in [httplib.UNAUTHORIZED, httplib.FORBIDDEN]:
            raise InvalidCredsError(self.body)
        elif self.status == httplib.MOVED_PERMANENTLY:
            bucket_region = self.headers.get('x-amz-bucket-region', None)
            used_region = self.connection.driver.region
            raise LibcloudError('This bucket is located in a different region. Please use the correct driver. Bucket region "%s", used region "%s".' % (bucket_region, used_region), driver=S3StorageDriver)
        raise LibcloudError('Unknown error. Status code: %d' % self.status, driver=S3StorageDriver)