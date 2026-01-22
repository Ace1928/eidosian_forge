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
def _get_region(self, request, url, imdsv2_session_token):
    """Retrieves the current AWS region from either the AWS_REGION or
        AWS_DEFAULT_REGION environment variable or from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            url (str): The AWS metadata server region URL.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            str: The current AWS region.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS region.
        """
    env_aws_region = os.environ.get(environment_vars.AWS_REGION)
    if env_aws_region is not None:
        return env_aws_region
    env_aws_region = os.environ.get(environment_vars.AWS_DEFAULT_REGION)
    if env_aws_region is not None:
        return env_aws_region
    if not self._region_url:
        raise exceptions.RefreshError('Unable to determine AWS region')
    headers = None
    if imdsv2_session_token is not None:
        headers = {'X-aws-ec2-metadata-token': imdsv2_session_token}
    response = request(url=self._region_url, method='GET', headers=headers)
    response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
    if response.status != 200:
        raise exceptions.RefreshError('Unable to retrieve AWS region', response_body)
    return response_body[:-1]