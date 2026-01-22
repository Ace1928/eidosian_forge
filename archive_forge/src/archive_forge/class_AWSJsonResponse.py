import hmac
import time
import base64
import hashlib
from typing import Dict, Type, Optional
from hashlib import sha256
from datetime import datetime
from libcloud.utils.py3 import ET, b, httplib, urlquote, basestring, _real_unicode
from libcloud.utils.xml import findall_ignore_namespace, findtext_ignore_namespace
from libcloud.common.base import BaseDriver, XmlResponse, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
class AWSJsonResponse(JsonResponse):
    """
    Amazon ECS response class.
    ECS API uses JSON unlike the s3, elb drivers
    """

    def parse_error(self):
        response = json.loads(self.body)
        code = response['__type']
        message = response.get('Message', response['message'])
        return '{}: {}'.format(code, message)