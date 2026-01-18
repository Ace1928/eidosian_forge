import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
Test refresh token with failure response.