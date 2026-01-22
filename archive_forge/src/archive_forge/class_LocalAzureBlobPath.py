import os
from ...cloudpath import CloudImplementation
from ...exceptions import MissingCredentialsError
from ..localclient import LocalClient
from ..localpath import LocalPath
class LocalAzureBlobPath(LocalPath):
    """Replacement for AzureBlobPath that uses the local file system. Intended as a monkeypatch
    substitute when writing tests.
    """
    cloud_prefix: str = 'az://'
    _cloud_meta = local_azure_blob_implementation

    @property
    def drive(self) -> str:
        return self.container

    def mkdir(self, parents=False, exist_ok=False):
        pass

    @property
    def container(self) -> str:
        return self._no_prefix.split('/', 1)[0]

    @property
    def blob(self) -> str:
        key = self._no_prefix_no_drive
        if key.startswith('/'):
            key = key[1:]
        return key

    @property
    def etag(self):
        return self.client._md5(self)

    @property
    def md5(self) -> str:
        return self.client._md5(self)