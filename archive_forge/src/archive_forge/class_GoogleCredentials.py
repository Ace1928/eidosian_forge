import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
class GoogleCredentials(OAuth2Credentials):
    """Application Default Credentials for use in calling Google APIs.

    The Application Default Credentials are being constructed as a function of
    the environment where the code is being run.
    More details can be found on this page:
    https://developers.google.com/accounts/docs/application-default-credentials

    Here is an example of how to use the Application Default Credentials for a
    service that requires authentication::

        from googleapiclient.discovery import build
        from oauth2client.client import GoogleCredentials

        credentials = GoogleCredentials.get_application_default()
        service = build('compute', 'v1', credentials=credentials)

        PROJECT = 'bamboo-machine-422'
        ZONE = 'us-central1-a'
        request = service.instances().list(project=PROJECT, zone=ZONE)
        response = request.execute()

        print(response)
    """
    NON_SERIALIZED_MEMBERS = frozenset(['_private_key']) | OAuth2Credentials.NON_SERIALIZED_MEMBERS
    "Members that aren't serialized when object is converted to JSON."

    def __init__(self, access_token, client_id, client_secret, refresh_token, token_expiry, token_uri, user_agent, revoke_uri=oauth2client.GOOGLE_REVOKE_URI):
        """Create an instance of GoogleCredentials.

        This constructor is not usually called by the user, instead
        GoogleCredentials objects are instantiated by
        GoogleCredentials.from_stream() or
        GoogleCredentials.get_application_default().

        Args:
            access_token: string, access token.
            client_id: string, client identifier.
            client_secret: string, client secret.
            refresh_token: string, refresh token.
            token_expiry: datetime, when the access_token expires.
            token_uri: string, URI of token endpoint.
            user_agent: string, The HTTP User-Agent to provide for this
                        application.
            revoke_uri: string, URI for revoke endpoint. Defaults to
                        oauth2client.GOOGLE_REVOKE_URI; a token can't be
                        revoked if this is None.
        """
        super(GoogleCredentials, self).__init__(access_token, client_id, client_secret, refresh_token, token_expiry, token_uri, user_agent, revoke_uri=revoke_uri)

    def create_scoped_required(self):
        """Whether this Credentials object is scopeless.

        create_scoped(scopes) method needs to be called in order to create
        a Credentials object for API calls.
        """
        return False

    def create_scoped(self, scopes):
        """Create a Credentials object for the given scopes.

        The Credentials type is preserved.
        """
        return self

    @classmethod
    def from_json(cls, json_data):
        from oauth2client import service_account
        data = json.loads(_helpers._from_bytes(json_data))
        if data['_module'] == 'oauth2client.service_account' and data['_class'] == 'ServiceAccountCredentials':
            return service_account.ServiceAccountCredentials.from_json(data)
        elif data['_module'] == 'oauth2client.service_account' and data['_class'] == '_JWTAccessCredentials':
            return service_account._JWTAccessCredentials.from_json(data)
        token_expiry = _parse_expiry(data.get('token_expiry'))
        google_credentials = cls(data['access_token'], data['client_id'], data['client_secret'], data['refresh_token'], token_expiry, data['token_uri'], data['user_agent'], revoke_uri=data.get('revoke_uri', None))
        google_credentials.invalid = data['invalid']
        return google_credentials

    @property
    def serialization_data(self):
        """Get the fields and values identifying the current credentials."""
        return {'type': 'authorized_user', 'client_id': self.client_id, 'client_secret': self.client_secret, 'refresh_token': self.refresh_token}

    @staticmethod
    def _implicit_credentials_from_gae():
        """Attempts to get implicit credentials in Google App Engine env.

        If the current environment is not detected as App Engine, returns None,
        indicating no Google App Engine credentials can be detected from the
        current environment.

        Returns:
            None, if not in GAE, else an appengine.AppAssertionCredentials
            object.
        """
        if not _in_gae_environment():
            return None
        return _get_application_default_credential_GAE()

    @staticmethod
    def _implicit_credentials_from_gce():
        """Attempts to get implicit credentials in Google Compute Engine env.

        If the current environment is not detected as Compute Engine, returns
        None, indicating no Google Compute Engine credentials can be detected
        from the current environment.

        Returns:
            None, if not in GCE, else a gce.AppAssertionCredentials object.
        """
        if not _in_gce_environment():
            return None
        return _get_application_default_credential_GCE()

    @staticmethod
    def _implicit_credentials_from_files():
        """Attempts to get implicit credentials from local credential files.

        First checks if the environment variable GOOGLE_APPLICATION_CREDENTIALS
        is set with a filename and then falls back to a configuration file (the
        "well known" file) associated with the 'gcloud' command line tool.

        Returns:
            Credentials object associated with the
            GOOGLE_APPLICATION_CREDENTIALS file or the "well known" file if
            either exist. If neither file is define, returns None, indicating
            no credentials from a file can detected from the current
            environment.
        """
        credentials_filename = _get_environment_variable_file()
        if not credentials_filename:
            credentials_filename = _get_well_known_file()
            if os.path.isfile(credentials_filename):
                extra_help = ' (produced automatically when running "gcloud auth login" command)'
            else:
                credentials_filename = None
        else:
            extra_help = ' (pointed to by ' + GOOGLE_APPLICATION_CREDENTIALS + ' environment variable)'
        if not credentials_filename:
            return
        SETTINGS.env_name = DEFAULT_ENV_NAME
        try:
            return _get_application_default_credential_from_file(credentials_filename)
        except (ApplicationDefaultCredentialsError, ValueError) as error:
            _raise_exception_for_reading_json(credentials_filename, extra_help, error)

    @classmethod
    def _get_implicit_credentials(cls):
        """Gets credentials implicitly from the environment.

        Checks environment in order of precedence:
        - Environment variable GOOGLE_APPLICATION_CREDENTIALS pointing to
          a file with stored credentials information.
        - Stored "well known" file associated with `gcloud` command line tool.
        - Google App Engine (production and testing)
        - Google Compute Engine production environment.

        Raises:
            ApplicationDefaultCredentialsError: raised when the credentials
                                                fail to be retrieved.
        """
        environ_checkers = [cls._implicit_credentials_from_files, cls._implicit_credentials_from_gae, cls._implicit_credentials_from_gce]
        for checker in environ_checkers:
            credentials = checker()
            if credentials is not None:
                return credentials
        raise ApplicationDefaultCredentialsError(ADC_HELP_MSG)

    @staticmethod
    def get_application_default():
        """Get the Application Default Credentials for the current environment.

        Raises:
            ApplicationDefaultCredentialsError: raised when the credentials
                                                fail to be retrieved.
        """
        return GoogleCredentials._get_implicit_credentials()

    @staticmethod
    def from_stream(credential_filename):
        """Create a Credentials object by reading information from a file.

        It returns an object of type GoogleCredentials.

        Args:
            credential_filename: the path to the file from where the
                                 credentials are to be read

        Raises:
            ApplicationDefaultCredentialsError: raised when the credentials
                                                fail to be retrieved.
        """
        if credential_filename and os.path.isfile(credential_filename):
            try:
                return _get_application_default_credential_from_file(credential_filename)
            except (ApplicationDefaultCredentialsError, ValueError) as error:
                extra_help = ' (provided as parameter to the from_stream() method)'
                _raise_exception_for_reading_json(credential_filename, extra_help, error)
        else:
            raise ApplicationDefaultCredentialsError('The parameter passed to the from_stream() method should point to a file.')