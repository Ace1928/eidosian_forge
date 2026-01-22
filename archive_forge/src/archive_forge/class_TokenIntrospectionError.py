from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from google.oauth2 import utils as oauth2_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from six.moves import http_client
from six.moves import urllib
class TokenIntrospectionError(Error):
    """Raised when an error is encountered while calling token introspection."""