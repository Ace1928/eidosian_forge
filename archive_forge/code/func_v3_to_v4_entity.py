from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_to_v4_entity(self, v3_entity, v4_entity):
    """Converts a v3 EntityProto to a v4 Entity.

    Args:
      v3_entity: an entity_pb.EntityProto
      v4_entity: an entity_v4_pb.Proto to populate
    """
    v4_entity.Clear()
    self.v3_to_v4_key(v3_entity.key(), v4_entity.mutable_key())
    if not v3_entity.key().has_app():
        v4_entity.clear_key()
    v4_properties = {}
    for v3_property in v3_entity.property_list():
        self.__add_v4_property_to_entity(v4_entity, v4_properties, v3_property, True)
    for v3_property in v3_entity.raw_property_list():
        self.__add_v4_property_to_entity(v4_entity, v4_properties, v3_property, False)