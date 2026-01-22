from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v1_string_property(self, entity, name, value, indexed):
    """Populates a single-string-valued v1 Property.

    Args:
      entity: the entity to populate
      name: the name of the property to populate
      value: the string value of the property
      indexed: whether the value should be indexed
    """
    v1_value = entity.properties[name]
    v1_value.exclude_from_indexes = not indexed
    v1_value.string_value = value