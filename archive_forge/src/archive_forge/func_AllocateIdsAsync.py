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
def AllocateIdsAsync(model_key, size=None, **kwargs):
    """Asynchronously allocates a range of IDs.

  Identical to datastore.AllocateIds() except returns an asynchronous object.
  Call get_result() on the return value to block on the call and get the
  results.
  """
    max = kwargs.pop('max', None)
    config = _GetConfigFromKwargs(kwargs)
    if getattr(config, 'read_policy', None) == EVENTUAL_CONSISTENCY:
        raise datastore_errors.BadRequestError('read_policy is only supported on read operations.')
    keys, _ = NormalizeAndTypeCheckKeys(model_key)
    if len(keys) > 1:
        raise datastore_errors.BadArgumentError('Cannot allocate IDs for more than one model key at a time')
    rpc = _GetConnection().async_allocate_ids(config, keys[0], size, max)
    return rpc