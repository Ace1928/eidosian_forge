from ...cloudpath import CloudImplementation
from ..localclient import LocalClient
from ..localpath import LocalPath
class LocalS3Path(LocalPath):
    """Replacement for S3Path that uses the local file system. Intended as a monkeypatch substitute
    when writing tests.
    """
    cloud_prefix: str = 's3://'
    _cloud_meta = local_s3_implementation

    @property
    def drive(self) -> str:
        return self.bucket

    def mkdir(self, parents=False, exist_ok=False):
        pass

    @property
    def bucket(self) -> str:
        return self._no_prefix.split('/', 1)[0]

    @property
    def key(self) -> str:
        key = self._no_prefix_no_drive
        if key.startswith('/'):
            key = key[1:]
        return key

    @property
    def etag(self):
        return self.client._md5(self)