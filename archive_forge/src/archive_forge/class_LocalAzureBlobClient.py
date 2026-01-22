import os
from ...cloudpath import CloudImplementation
from ...exceptions import MissingCredentialsError
from ..localclient import LocalClient
from ..localpath import LocalPath
class LocalAzureBlobClient(LocalClient):
    """Replacement for AzureBlobClient that uses the local file system. Intended as a monkeypatch
    substitute when writing tests.
    """
    _cloud_meta = local_azure_blob_implementation

    def __init__(self, *args, **kwargs):
        cred_opts = [kwargs.get('blob_service_client', None), kwargs.get('connection_string', None), kwargs.get('account_url', None), os.getenv('AZURE_STORAGE_CONNECTION_STRING', None)]
        super().__init__(*args, **kwargs)
        if all((opt is None for opt in cred_opts)):
            raise MissingCredentialsError('AzureBlobClient does not support anonymous instantiation. Credentials are required; see docs for options.')