import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
def _MakeSyncCall(service, call, request, response, config=None):
    """The APIProxy entry point for a synchronous API call.

  Args:
    service: For backwards compatibility, must be 'datastore_v3'.
    call: String representing which function to call.
    request: Protocol buffer for the request.
    response: Protocol buffer for the response.
    config: Optional Configuration to use for this request.

  Returns:
    Response protocol buffer. Caller should always use returned value
    which may or may not be same as passed in 'response'.

  Raises:
    apiproxy_errors.Error or a subclass.
  """
    conn = _GetConnection()
    if isinstance(request, datastore_pb.Query):
        conn._set_request_read_policy(request, config)
        conn._set_request_transaction(request)
    rpc = conn._make_rpc_call(config, call, request, response)
    conn.check_rpc_success(rpc)
    return response