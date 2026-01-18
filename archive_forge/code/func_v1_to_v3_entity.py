from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v1_to_v3_entity(self, v1_entity, v3_entity, is_projection=False):
    """Converts a v1 Entity to a v3 EntityProto.

    Args:
      v1_entity: an googledatastore.Entity
      v3_entity: an entity_pb.EntityProto to populate
      is_projection: True if the v1_entity is from a projection query.
    """
    v3_entity.Clear()
    for property_name, v1_value in six.iteritems(v1_entity.properties):
        if v1_value.HasField('array_value'):
            if len(v1_value.array_value.values) == 0:
                empty_list = self.__new_v3_property(v3_entity, not v1_value.exclude_from_indexes)
                empty_list.set_name(property_name.encode('utf-8'))
                empty_list.set_multiple(False)
                empty_list.set_meaning(MEANING_EMPTY_LIST)
                empty_list.mutable_value()
            else:
                for v1_sub_value in v1_value.array_value.values:
                    list_element = self.__new_v3_property(v3_entity, not v1_sub_value.exclude_from_indexes)
                    self.v1_to_v3_property(property_name, True, is_projection, v1_sub_value, list_element)
        else:
            value_property = self.__new_v3_property(v3_entity, not v1_value.exclude_from_indexes)
            self.v1_to_v3_property(property_name, False, is_projection, v1_value, value_property)
    if v1_entity.HasField('key'):
        v1_key = v1_entity.key
        self.v1_to_v3_reference(v1_key, v3_entity.mutable_key())
        v3_ref = v3_entity.key()
        self.v3_reference_to_group(v3_ref, v3_entity.mutable_entity_group())
    else:
        pass