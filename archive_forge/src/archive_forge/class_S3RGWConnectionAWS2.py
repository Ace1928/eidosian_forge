from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, SignedAWSConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.drivers.s3 import (
class S3RGWConnectionAWS2(S3Connection):

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, proxy_url=None, token=None, retry_delay=None, backoff=None, **kwargs):
        super().__init__(user_id, key, secure, host, port, url, timeout, proxy_url, token, retry_delay, backoff)