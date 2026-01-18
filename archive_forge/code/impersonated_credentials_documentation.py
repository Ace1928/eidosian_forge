import base64
import copy
from datetime import datetime
import json
import six
from six.moves import http_client
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt

        Args:
            target_credentials (google.auth.Credentials): The target
                credential used as to acquire the id tokens for.
            target_audience (string): Audience to issue the token for.
            include_email (bool): Include email in IdToken
            quota_project_id (Optional[str]):  The project ID used for
                quota and billing.
        