from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_to_v1_entity(self, v3_entity, v1_entity):
    """Converts a v3 EntityProto to a v1 Entity.

    Args:
      v3_entity: an entity_pb.EntityProto
      v1_entity: an googledatastore.Proto to populate
    """
    v1_entity.Clear()
    self.v3_to_v1_key(v3_entity.key(), v1_entity.key)
    if not v3_entity.key().has_app():
        v1_entity.ClearField('key')
    for v3_property in v3_entity.property_list():
        self.__add_v1_property_to_entity(v1_entity, v3_property, True)
    for v3_property in v3_entity.raw_property_list():
        self.__add_v1_property_to_entity(v1_entity, v3_property, False)