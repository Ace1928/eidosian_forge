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
def _extract_entity_group(self, value):
    """Internal helper: extracts the entity group from a key or entity.

    Supports both v3 and v1 protobufs.

    Args:
      value: an entity_pb.{Reference, EntityProto} or
          googledatastore.{Key, Entity}.

    Returns:
      A tuple consisting of:
        - kind
        - name, id, or ('new', unique id)
    """
    if _CLOUD_DATASTORE_ENABLED and isinstance(value, googledatastore.Entity):
        value = value.key
    if isinstance(value, entity_pb.EntityProto):
        value = value.key()
    if _CLOUD_DATASTORE_ENABLED and isinstance(value, googledatastore.Key):
        elem = value.path[0]
        elem_id = elem.id
        elem_name = elem.name
        kind = elem.kind
    else:
        elem = value.path().element(0)
        kind = elem.type()
        elem_id = elem.id()
        elem_name = elem.name()
    return (kind, elem_id or elem_name or ('new', id(elem)))