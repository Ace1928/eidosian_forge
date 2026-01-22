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
def _AddHeaders(headers_func):
    """Returns a function that adds headers to client call details."""
    headers = headers_func()

    def AddHeaders(client_call_details):
        if not headers:
            return client_call_details
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        for header, value in headers:
            metadata.append((header.lower(), value))
        new_client_call_details = _ClientCallDetails(client_call_details.method, client_call_details.timeout, metadata, client_call_details.credentials, client_call_details.wait_for_ready, client_call_details.compression)
        return new_client_call_details
    return AddHeaders