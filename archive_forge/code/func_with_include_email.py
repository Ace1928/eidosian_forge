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
def with_include_email(self, include_email):
    return self.__class__(target_credentials=self._target_credentials, target_audience=self._target_audience, include_email=include_email, quota_project_id=self._quota_project_id)