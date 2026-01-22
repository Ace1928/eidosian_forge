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
@staticmethod
def _FromPb(pb, require_valid_key=True, default_kind='<not specified>'):
    """Static factory method. Returns the Entity representation of the
    given protocol buffer (datastore_pb.Entity). Not intended to be used by
    application developers.

    The Entity PB's key must be complete. If it isn't, an AssertionError is
    raised.

    Args:
      # a protocol buffer Entity
      pb: datastore_pb.Entity
      default_kind: str, the kind to use if the pb has no key.

    Returns:
      # the Entity representation of the argument
      Entity
    """
    if not pb.key().path().element_size():
        pb.mutable_key().CopyFrom(Key.from_path(default_kind, 0)._ToPb())
    last_path = pb.key().path().element_list()[-1]
    if require_valid_key:
        assert last_path.has_id() ^ last_path.has_name()
        if last_path.has_id():
            assert last_path.id() != 0
        else:
            assert last_path.has_name()
            assert last_path.name()
    unindexed_properties = [unicode(p.name(), 'utf-8') for p in pb.raw_property_list()]
    if pb.key().has_name_space():
        namespace = pb.key().name_space()
    else:
        namespace = ''
    e = Entity(unicode(last_path.type(), 'utf-8'), unindexed_properties=unindexed_properties, _app=pb.key().app(), namespace=namespace)
    ref = e.__key._Key__reference
    ref.CopyFrom(pb.key())
    temporary_values = {}
    for prop_list in (pb.property_list(), pb.raw_property_list()):
        for prop in prop_list:
            if prop.meaning() == entity_pb.Property.INDEX_VALUE:
                e.__projection = True
            try:
                value = datastore_types.FromPropertyPb(prop)
            except (AssertionError, AttributeError, TypeError, ValueError):
                raise datastore_errors.Error('Property %s is corrupt in the datastore:\n%s' % (prop.name(), traceback.format_exc()))
            multiple = prop.multiple()
            if multiple:
                value = [value]
            name = prop.name()
            cur_value = temporary_values.get(name)
            if cur_value is None:
                temporary_values[name] = value
            elif not multiple or not isinstance(cur_value, list):
                raise datastore_errors.Error('Property %s is corrupt in the datastore; it has multiple values, but is not marked as multiply valued.' % name)
            else:
                cur_value.extend(value)
    for name, value in temporary_values.iteritems():
        decoded_name = unicode(name, 'utf-8')
        datastore_types.ValidateReadProperty(decoded_name, value)
        dict.__setitem__(e, decoded_name, value)
    return e