from libcloud.common.aws import SignedAWSConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.drivers.s3 import S3Connection, BaseS3Connection, BaseS3StorageDriver
class DigitalOceanSpacesStorageDriver(BaseS3StorageDriver):
    name = 'DigitalOcean Spaces'
    website = 'https://www.digitalocean.com/products/object-storage/'
    supports_chunked_encoding = False
    supports_s3_multipart_upload = True

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=DO_SPACES_DEFAULT_REGION, **kwargs):
        if region not in DO_SPACES_HOSTS_BY_REGION:
            raise LibcloudError('Unknown region (%s)' % region, driver=self)
        host = DO_SPACES_HOSTS_BY_REGION[region]
        self.name = 'DigitalOcean Spaces (%s)' % region
        self.region_name = region
        self.signature_version = str(kwargs.pop('signature_version', DEFAULT_SIGNATURE_VERSION))
        if self.signature_version == '2':
            self.connectionCls = DOSpacesConnectionAWS2
        elif self.signature_version == '4':
            self.connectionCls = DOSpacesConnectionAWS4
        else:
            raise ValueError('Invalid signature_version: %s' % self.signature_version)
        self.connectionCls.host = host
        super().__init__(key, secret, secure, host, port, api_version, region, **kwargs)

    def _ex_connection_class_kwargs(self):
        kwargs = {}
        kwargs['signature_version'] = self.signature_version
        return kwargs