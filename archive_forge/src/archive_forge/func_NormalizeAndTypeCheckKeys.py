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
def NormalizeAndTypeCheckKeys(keys):
    """Normalizes and type checks that the given argument is a valid key or keys.

  A wrapper around NormalizeAndTypeCheck() that accepts strings, Keys, and
  Entities, and normalizes to Keys.

  Args:
    keys: a Key or sequence of Keys

  Returns:
    A (list of Keys, bool) tuple. See NormalizeAndTypeCheck.

  Raises:
    BadArgumentError: arg is not an instance or sequence of one of the given
    types.
  """
    keys, multiple = NormalizeAndTypeCheck(keys, (basestring, Entity, Key))
    keys = [_GetCompleteKeyOrError(key) for key in keys]
    return (keys, multiple)