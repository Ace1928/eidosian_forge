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
def ShouldRecoverFromQuotaProject(credentials):
    """Returns a callback for handling Quota Project fallback."""
    if not base.UserProjectQuotaWithFallbackEnabled():
        return lambda _: False

    def _ShouldRecover(response):
        if response.code() != grpc.StatusCode.PERMISSION_DENIED:
            return False
        if IsUserProjectError(response.trailing_metadata()):
            credentials._quota_project_id = None
            return True
        return False
    return _ShouldRecover