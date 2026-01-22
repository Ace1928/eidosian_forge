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
class QuotaProjectInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor):
    """API Enablement Interceptor for prompting to enable APIs."""

    def __init__(self, credentials):
        self.credentials = credentials

    def intercept_call(self, continuation, client_call_details, request):
        response = continuation(client_call_details, request)
        if response.code() != grpc.StatusCode.PERMISSION_DENIED:
            return response
        if not IsUserProjectError(response.trailing_metadata()):
            return response
        quota_project = self.credentials._quota_project_id
        self.credentials._quota_project_id = None
        try:
            return continuation(client_call_details, request)
        finally:
            self.credentials._quota_project_id = quota_project

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercepts a unary-unary invocation asynchronously."""
        return self.intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        """Intercepts a stream-unary invocation asynchronously."""
        return self.intercept_call(continuation, client_call_details, request_iterator)