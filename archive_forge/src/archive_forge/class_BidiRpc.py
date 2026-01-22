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
class BidiRpc(bidi.ResumableBidiRpc):
    """Bidi implementation to be used throughout codebase."""

    def __init__(self, client, start_rpc, initial_request=None):
        """Initializes a BidiRpc instances.

    Args:
        client: GAPIC Wrapper client to use.
        start_rpc (grpc.StreamStreamMultiCallable): The gRPC method used to
            start the RPC.
        initial_request: The initial request to
            yield. This is useful if an initial request is needed to start the
            stream.
    """
        super(BidiRpc, self).__init__(start_rpc, initial_request=initial_request, should_recover=ShouldRecover(client.credentials))