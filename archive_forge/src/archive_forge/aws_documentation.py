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
Creates an AWS Credentials instance from an external account json file.

        Args:
            filename (str): The path to the AWS external account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.aws.Credentials: The constructed credentials.
        