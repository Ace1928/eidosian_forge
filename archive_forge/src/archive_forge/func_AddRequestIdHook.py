from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def AddRequestIdHook(unused_ref, unused_args, request):
    """Fills a unique identifier for a request with a requestId field."""
    request.requestId = request_utils.GenerateRequestId()
    return request