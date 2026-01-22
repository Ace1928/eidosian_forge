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
class BearerAuth(TokenSigner):
    """
    Performs bearer token authorization by placing the bearer token in the
    Authorization header as specified by Section 2.1 of RFC 6750.

    https://datatracker.ietf.org/doc/html/rfc6750#section-2.1
    """

    def add_auth(self, request):
        if self.auth_token is None:
            raise NoAuthTokenError()
        auth_header = f'Bearer {self.auth_token.token}'
        if 'Authorization' in request.headers:
            del request.headers['Authorization']
        request.headers['Authorization'] = auth_header