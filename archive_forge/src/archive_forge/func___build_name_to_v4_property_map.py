from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __build_name_to_v4_property_map(self, v4_entity):
    property_map = {}
    for prop in v4_entity.property_list():
        property_map[prop.name()] = prop
    return property_map