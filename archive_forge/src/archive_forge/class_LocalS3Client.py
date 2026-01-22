from ...cloudpath import CloudImplementation
from ..localclient import LocalClient
from ..localpath import LocalPath
class LocalS3Client(LocalClient):
    """Replacement for S3Client that uses the local file system. Intended as a monkeypatch
    substitute when writing tests.
    """
    _cloud_meta = local_s3_implementation