import base64
import urllib
import requests
import requests.exceptions
from requests.adapters import HTTPAdapter, Retry
from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile
def _create_handle(self, path, overwrite=True):
    """
        Internal function to create a handle, which can be used to
        write blocks of a file to DBFS.
        A handle has a unique identifier which needs to be passed
        whenever written during this transaction.
        The handle is active for 10 minutes - after that a new
        write transaction needs to be created.
        Make sure to close the handle after you are finished.

        Parameters
        ----------
        path: str
            Absolute path for this file.
        overwrite: bool
            If a file already exist at this location, either overwrite
            it or raise an exception.
        """
    try:
        r = self._send_to_api(method='post', endpoint='create', json={'path': path, 'overwrite': overwrite})
        return r['handle']
    except DatabricksException as e:
        if e.error_code == 'RESOURCE_ALREADY_EXISTS':
            raise FileExistsError(e.message)
        raise e