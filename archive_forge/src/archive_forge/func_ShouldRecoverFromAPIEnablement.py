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
def ShouldRecoverFromAPIEnablement():
    """Returns a callback for checking API enablement errors."""
    state = {'already_prompted_to_enable': False, 'api_enabled': False}

    def _ShouldRecover(response):
        if response.code() != grpc.StatusCode.PERMISSION_DENIED:
            return False
        enablement_info = api_enablement.GetApiEnablementInfo(response.details())
        if enablement_info:
            if state['already_prompted_to_enable']:
                return state['api_enabled']
            state['already_prompted_to_enable'] = True
            api_enable_attempted = api_enablement.PromptToEnableApi(*enablement_info)
            if api_enable_attempted:
                state['api_enabled'] = api_enable_attempted
                return True
        return False
    return _ShouldRecover