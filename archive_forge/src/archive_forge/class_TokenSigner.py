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
class TokenSigner(BaseSigner):
    REQUIRES_TOKEN = True
    '\n    Signers that expect an authorization token to perform the authorization\n    '

    def __init__(self, auth_token):
        self.auth_token = auth_token