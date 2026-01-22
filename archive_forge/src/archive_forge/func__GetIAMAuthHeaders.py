from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from google.api_core import bidi
from google.rpc import error_details_pb2
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.calliope import base
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_proxy_types
import grpc
from six.moves import urllib
import socks
def _GetIAMAuthHeaders():
    """Returns the IAM authorization headers to be used."""
    headers = []
    authority_selector = properties.VALUES.auth.authority_selector.Get()
    if authority_selector:
        headers.append((IAM_AUTHORITY_SELECTOR_HEADER, authority_selector))
    authorization_token = None
    authorization_token_file = properties.VALUES.auth.authorization_token_file.Get()
    if authorization_token_file:
        try:
            authorization_token = files.ReadFileContents(authorization_token_file)
        except files.Error as e:
            raise Error(e)
    if authorization_token:
        headers.append((IAM_AUTHORIZATION_TOKEN_HEADER, authorization_token.strip()))
    return headers