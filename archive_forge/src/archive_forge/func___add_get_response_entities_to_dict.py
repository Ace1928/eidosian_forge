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
def __add_get_response_entities_to_dict(self, get_response, result_dict):
    """Converts entities from the get response and adds them to the dict.

    The Key for the dict will be calculated via
    datastore_types.ReferenceToKeyValue.  There will be no entry for entities
    that were not found.

    Args:
      get_response: A datastore_pb.GetResponse or
          googledatastore.LookupResponse.
      result_dict: The dict to add results to.
    """
    if _CLOUD_DATASTORE_ENABLED and isinstance(get_response, googledatastore.LookupResponse):
        for result in get_response.found:
            v1_key = result.entity.key
            entity = self.__adapter.pb_v1_to_entity(result.entity, False)
            result_dict[datastore_types.ReferenceToKeyValue(v1_key)] = entity
    else:
        for entity_result in get_response.entity_list():
            if entity_result.has_entity():
                reference_pb = entity_result.entity().key()
                hashable_key = datastore_types.ReferenceToKeyValue(reference_pb)
                entity = self.__adapter.pb_to_entity(entity_result.entity())
                result_dict[hashable_key] = entity