from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __add_v4_property_to_entity(self, v4_entity, property_map, v3_property, indexed):
    """Adds a v4 Property to an entity or modifies an existing one.

    property_map is used to track of properties that have already been added.
    The same dict should be used for all of an entity's properties.

    Args:
      v4_entity: an entity_v4_pb.Entity
      property_map: a dict of name -> v4_property
      v3_property: an entity_pb.Property to convert to v4 and add to the dict
      indexed: whether the property is indexed
    """
    property_name = v3_property.name()
    if property_name in property_map:
        v4_property = property_map[property_name]
    else:
        v4_property = v4_entity.add_property()
        v4_property.set_name(property_name)
        property_map[property_name] = v4_property
    if v3_property.multiple():
        self.v3_property_to_v4_value(v3_property, indexed, v4_property.mutable_value().add_list_value())
    else:
        self.v3_property_to_v4_value(v3_property, indexed, v4_property.mutable_value())