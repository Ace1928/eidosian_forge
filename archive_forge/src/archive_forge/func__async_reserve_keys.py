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
def _async_reserve_keys(self, config, keys, extra_hook=None):
    """Asynchronous AllocateIds operation to reserve the given keys.

    Sends one or more v3 AllocateIds rpcs with keys to reserve.
    Reserved keys must be complete and must have valid ids.

    Args:
      config: A Configuration object or None to use Connection default.
      keys: Iterable of user-level keys.
      extra_hook: Optional function to be called on rpc result.

    Returns:
      None, or the result of user-supplied extra_hook.
    """

    def to_id_key(key):
        if key.path().element_size() == 1:
            return 'root_idkey'
        else:
            return self._extract_entity_group(key)
    keys_by_idkey = self._map_and_group(keys, self.__adapter.key_to_pb, to_id_key)
    max_count = Configuration.max_allocate_ids_keys(config, self.__config) or self.MAX_ALLOCATE_IDS_KEYS
    rpcs = []
    pbsgen = self._generate_pb_lists(keys_by_idkey, 0, max_count, None, config)
    for pbs, _ in pbsgen:
        req = datastore_pb.AllocateIdsRequest()
        req.reserve_list().extend(pbs)
        resp = datastore_pb.AllocateIdsResponse()
        rpcs.append(self._make_rpc_call(config, 'AllocateIds', req, resp, get_result_hook=self.__reserve_keys_hook, user_data=extra_hook, service_name=_DATASTORE_V3))
    return MultiRpc(rpcs)