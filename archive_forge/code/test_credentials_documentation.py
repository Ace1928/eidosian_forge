import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
 Test that it is possible to refresh credentials
        generated from `with_quota_project`.

        Instead of mocking the methods, the HTTP responses
        have been mocked.
        