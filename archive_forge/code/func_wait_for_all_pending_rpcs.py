from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def wait_for_all_pending_rpcs(self):
    """Wait for all currently pending RPCs to complete."""
    while self.__pending_rpcs:
        try:
            rpc = apiproxy_stub_map.UserRPC.wait_any(self.__pending_rpcs)
        except Exception:
            logging.info('wait_for_all_pending_rpcs(): exception in wait_any()', exc_info=True)
            continue
        if rpc is None:
            logging.debug('wait_any() returned None')
            continue
        assert rpc.state == apiproxy_rpc.RPC.FINISHING
        if rpc in self.__pending_rpcs:
            try:
                self.check_rpc_success(rpc)
            except Exception:
                logging.info('wait_for_all_pending_rpcs(): exception in check_rpc_success()', exc_info=True)