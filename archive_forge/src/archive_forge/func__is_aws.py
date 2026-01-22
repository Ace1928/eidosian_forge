import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@classmethod
def _is_aws(cls):
    try:
        r = cls.http.request('PUT', AWS_TOKEN_URL, headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
        if r.status != 200:
            return False
        cls.aws_token = r.data.decode()
        return True
    except Exception:
        return False