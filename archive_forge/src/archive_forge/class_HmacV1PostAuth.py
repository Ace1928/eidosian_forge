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
class HmacV1PostAuth(HmacV1Auth):
    """
    Generates a presigned post for s3.

    Spec from this document:

    http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingHTTPPOST.html
    """

    def add_auth(self, request):
        fields = {}
        if request.context.get('s3-presign-post-fields', None) is not None:
            fields = request.context['s3-presign-post-fields']
        policy = {}
        conditions = []
        if request.context.get('s3-presign-post-policy', None) is not None:
            policy = request.context['s3-presign-post-policy']
            if policy.get('conditions', None) is not None:
                conditions = policy['conditions']
        policy['conditions'] = conditions
        fields['AWSAccessKeyId'] = self.credentials.access_key
        if self.credentials.token is not None:
            fields['x-amz-security-token'] = self.credentials.token
            conditions.append({'x-amz-security-token': self.credentials.token})
        fields['policy'] = base64.b64encode(json.dumps(policy).encode('utf-8')).decode('utf-8')
        fields['signature'] = self.sign_string(fields['policy'])
        request.context['s3-presign-post-fields'] = fields
        request.context['s3-presign-post-policy'] = policy