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
def _get_metadata_role_name(self, request, imdsv2_session_token):
    """Retrieves the AWS role currently attached to the current AWS
        workload by querying the AWS metadata server. This is needed for the
        AWS metadata server security credentials endpoint in order to retrieve
        the AWS security credentials needed to sign requests to AWS APIs.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            str: The AWS role name.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS role name.
        """
    if self._security_credentials_url is None:
        raise exceptions.RefreshError('Unable to determine the AWS metadata server security credentials endpoint')
    headers = None
    if imdsv2_session_token is not None:
        headers = {'X-aws-ec2-metadata-token': imdsv2_session_token}
    response = request(url=self._security_credentials_url, method='GET', headers=headers)
    response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
    if response.status != http_client.OK:
        raise exceptions.RefreshError('Unable to retrieve AWS role name', response_body)
    return response_body