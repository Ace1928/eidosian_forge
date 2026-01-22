import base64
import calendar
import datetime
import functools
import hmac
import json
import logging
import time
from collections.abc import Mapping
from email.utils import formatdate
from hashlib import sha1, sha256
from operator import itemgetter
from botocore.compat import (
from botocore.exceptions import NoAuthTokenError, NoCredentialsError
from botocore.utils import (
from botocore.compat import MD5_AVAILABLE  # noqa
class SigV3Auth(BaseSigner):

    def __init__(self, credentials):
        self.credentials = credentials

    def add_auth(self, request):
        if self.credentials is None:
            raise NoCredentialsError()
        if 'Date' in request.headers:
            del request.headers['Date']
        request.headers['Date'] = formatdate(usegmt=True)
        if self.credentials.token:
            if 'X-Amz-Security-Token' in request.headers:
                del request.headers['X-Amz-Security-Token']
            request.headers['X-Amz-Security-Token'] = self.credentials.token
        new_hmac = hmac.new(self.credentials.secret_key.encode('utf-8'), digestmod=sha256)
        new_hmac.update(request.headers['Date'].encode('utf-8'))
        encoded_signature = encodebytes(new_hmac.digest()).strip()
        signature = f'AWS3-HTTPS AWSAccessKeyId={self.credentials.access_key},Algorithm=HmacSHA256,Signature={encoded_signature.decode('utf-8')}'
        if 'X-Amzn-Authorization' in request.headers:
            del request.headers['X-Amzn-Authorization']
        request.headers['X-Amzn-Authorization'] = signature