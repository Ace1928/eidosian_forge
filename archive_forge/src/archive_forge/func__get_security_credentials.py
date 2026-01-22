import hashlib
import hmac
import json
import os
import posixpath
import re
from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
def _get_security_credentials(self, request, imdsv2_session_token):
    """Retrieves the AWS security credentials required for signing AWS
        requests from either the AWS security credentials environment variables
        or from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            Mapping[str, str]: The AWS security credentials dictionary object.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS security credentials.
        """
    env_aws_access_key_id = os.environ.get(environment_vars.AWS_ACCESS_KEY_ID)
    env_aws_secret_access_key = os.environ.get(environment_vars.AWS_SECRET_ACCESS_KEY)
    env_aws_session_token = os.environ.get(environment_vars.AWS_SESSION_TOKEN)
    if env_aws_access_key_id and env_aws_secret_access_key:
        return {'access_key_id': env_aws_access_key_id, 'secret_access_key': env_aws_secret_access_key, 'security_token': env_aws_session_token}
    role_name = self._get_metadata_role_name(request, imdsv2_session_token)
    credentials = self._get_metadata_security_credentials(request, role_name, imdsv2_session_token)
    return {'access_key_id': credentials.get('AccessKeyId'), 'secret_access_key': credentials.get('SecretAccessKey'), 'security_token': credentials.get('Token')}