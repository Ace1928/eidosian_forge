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
def CreateTransactionOptions(**kwds):
    """Create a configuration object for use in configuring transactions.

  This configuration can be passed as run_in_transaction_option's first
  argument.

  Args:
    deadline: Optional deadline; default None (which means the
      system default deadline will be used, typically 5 seconds).
    on_completion: Optional callback function; default None.  If
      specified, it will be called with a UserRPC object as argument
      when an RPC completes.
    xg: set to true to allow cross-group transactions (high replication
      datastore only)
    retries: set the number of retries for a transaction
    **kwds: Other keyword arguments as long as they are supported by
      datastore_rpc.TransactionOptions().

  Returns:
    A datastore_rpc.TransactionOptions instance.
  """
    return datastore_rpc.TransactionOptions(**kwds)