import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
class AWSSession(Session):
    """Configures access to secured resources stored in AWS S3.
    """

    def __init__(self, session=None, aws_unsigned=None, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None, region_name=None, profile_name=None, endpoint_url=None, requester_pays=False):
        """Create a new AWS session

        Parameters
        ----------
        session : optional
            A boto3 session object.
        aws_unsigned : bool, optional (default: False)
            If True, requests will be unsigned.
        aws_access_key_id : str, optional
            An access key id, as per boto3.
        aws_secret_access_key : str, optional
            A secret access key, as per boto3.
        aws_session_token : str, optional
            A session token, as per boto3.
        region_name : str, optional
            A region name, as per boto3.
        profile_name : str, optional
            A shared credentials profile name, as per boto3.
        endpoint_url: str, optional
            An endpoint_url, as per GDAL's AWS_S3_ENPOINT
        requester_pays : bool, optional
            True if the requester agrees to pay transfer costs (default:
            False)
        """
        if aws_unsigned is None:
            aws_unsigned = parse_bool(os.getenv('AWS_NO_SIGN_REQUEST', False))
        if session:
            self._session = session
        elif aws_unsigned:
            self._session = SimpleNamespace(region_name=region_name)
        else:
            self._session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=region_name, profile_name=profile_name)
        self.requester_pays = requester_pays
        self.unsigned = aws_unsigned
        self.endpoint_url = endpoint_url
        self._creds = self._session.get_credentials() if not self.unsigned else None

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
        return 'AWS_ACCESS_KEY_ID' in config and 'AWS_SECRET_ACCESS_KEY' in config or 'AWS_NO_SIGN_REQUEST' in config

    @property
    def credentials(self):
        """The session credentials as a dict"""
        res = {}
        if self._creds:
            frozen_creds = self._creds.get_frozen_credentials()
            if frozen_creds.access_key:
                res['aws_access_key_id'] = frozen_creds.access_key
            if frozen_creds.secret_key:
                res['aws_secret_access_key'] = frozen_creds.secret_key
            if frozen_creds.token:
                res['aws_session_token'] = frozen_creds.token
        if self._session.region_name:
            res['aws_region'] = self._session.region_name
        if self.requester_pays:
            res['aws_request_payer'] = 'requester'
        if self.endpoint_url:
            res['aws_s3_endpoint'] = self.endpoint_url
        return res

    def get_credential_options(self):
        """Get credentials as GDAL configuration options

        Returns
        -------
        dict

        """
        if self.unsigned:
            opts = {'AWS_NO_SIGN_REQUEST': 'YES'}
            opts.update({k.upper(): v for k, v in self.credentials.items() if k in ('aws_region', 'aws_s3_endpoint')})
            return opts
        else:
            return {k.upper(): v for k, v in self.credentials.items()}