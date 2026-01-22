import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
class SwiftSession(Session):
    """Configures access to secured resources stored in OpenStack Swift Object Storage.
    """

    def __init__(self, session=None, swift_storage_url=None, swift_auth_token=None, swift_auth_v1_url=None, swift_user=None, swift_key=None):
        """Create new OpenStack Swift Object Storage Session.

        Three methods are possible:
            1. Create session by the swiftclient library.
            2. The SWIFT_STORAGE_URL and SWIFT_AUTH_TOKEN (this method is recommended by GDAL docs).
            3. The SWIFT_AUTH_V1_URL, SWIFT_USER and SWIFT_KEY (This depends on the swiftclient library).

        Parameters
        ----------
        session: optional
            A swiftclient connection object
        swift_storage_url:
            the storage URL
        swift_auth_token:
            the value of the x-auth-token authorization token
        swift_storage_url: string, optional
            authentication URL
        swift_user: string, optional
            user name to authenticate as
        swift_key: string, optional
            key/password to authenticate with

        Examples
        --------
        >>> import rasterio
        >>> from rasterio.session import SwiftSession
        >>> fp = '/vsiswift/bucket/key.tif'
        >>> conn = Connection(authurl='http://127.0.0.1:7777/auth/v1.0', user='test:tester', key='testing')
        >>> session = SwiftSession(conn)
        >>> with rasterio.Env(session):
        >>>     with rasterio.open(fp) as src:
        >>>         print(src.profile)

        """
        if swift_storage_url and swift_auth_token:
            self._creds = {'swift_storage_url': swift_storage_url, 'swift_auth_token': swift_auth_token}
        else:
            from swiftclient.client import Connection
            if session:
                self._session = session
            else:
                self._session = Connection(authurl=swift_auth_v1_url, user=swift_user, key=swift_key)
            self._creds = {'swift_storage_url': self._session.get_auth()[0], 'swift_auth_token': self._session.get_auth()[1]}

    @classmethod
    def hascreds(cls, config):
        """Determine if the given configuration has proper credentials
        Parameters
        ----------
        cls : class
            A Session class.
        config : dict
            GDAL configuration as a dict.
        Returns
        -------
        bool
        """
        return 'SWIFT_STORAGE_URL' in config and 'SWIFT_AUTH_TOKEN' in config

    @property
    def credentials(self):
        """The session credentials as a dict"""
        return self._creds

    def get_credential_options(self):
        """Get credentials as GDAL configuration options
        Returns
        -------
        dict
        """
        return {k.upper(): v for k, v in self.credentials.items()}