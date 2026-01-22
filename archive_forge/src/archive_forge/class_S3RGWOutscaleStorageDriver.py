from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, SignedAWSConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.drivers.s3 import (
class S3RGWOutscaleStorageDriver(S3RGWStorageDriver):
    name = 'RGW Outscale'
    website = 'https://en.outscale.com/'

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=S3_RGW_OUTSCALE_DEFAULT_REGION, **kwargs):
        if region not in S3_RGW_OUTSCALE_HOSTS_BY_REGION:
            raise LibcloudError('Unknown region (%s)' % region, driver=self)
        host = S3_RGW_OUTSCALE_HOSTS_BY_REGION[region]
        kwargs['name'] = 'OUTSCALE Ceph RGW S3 (%s)' % region
        super().__init__(key, secret, secure, host, port, api_version, region, **kwargs)