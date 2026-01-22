import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
class GSSession(Session):
    """Configures access to secured resources stored in Google Cloud Storage
    """

    def __init__(self, google_application_credentials=None):
        """Create new Google Cloude Storage session

        Parameters
        ----------
        google_application_credentials: string
            Path to the google application credentials JSON file.
        """
        if google_application_credentials is not None:
            self._creds = {'google_application_credentials': google_application_credentials}
        else:
            self._creds = {}

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
        return 'GOOGLE_APPLICATION_CREDENTIALS' in config

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