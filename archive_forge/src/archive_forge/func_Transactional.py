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
def Transactional(_func=None, **kwargs):
    """A decorator that makes sure a function is run in a transaction.

  Defaults propagation to datastore_rpc.TransactionOptions.ALLOWED, which means
  any existing transaction will be used in place of creating a new one.

  WARNING: Reading from the datastore while in a transaction will not see any
  changes made in the same transaction. If the function being decorated relies
  on seeing all changes made in the calling scoope, set
  propagation=datastore_rpc.TransactionOptions.NESTED.

  Args:
    _func: do not use.
    **kwargs: TransactionOptions configuration options.

  Returns:
    A wrapper for the given function that creates a new transaction if needed.
  """
    if _func is not None:
        return Transactional()(_func)
    if not kwargs.pop('require_new', None):
        kwargs.setdefault('propagation', datastore_rpc.TransactionOptions.ALLOWED)
    options = datastore_rpc.TransactionOptions(**kwargs)

    def outer_wrapper(func):

        def inner_wrapper(*args, **kwds):
            return RunInTransactionOptions(options, func, *args, **kwds)
        return inner_wrapper
    return outer_wrapper