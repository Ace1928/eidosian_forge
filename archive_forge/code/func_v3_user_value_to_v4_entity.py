from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_user_value_to_v4_entity(self, v3_user_value, v4_entity):
    """Converts a v3 UserValue to a v4 user Entity.

    Args:
      v3_user_value: an entity_pb.Property_UserValue
      v4_entity: an entity_v4_pb.Entity to populate
    """
    v4_entity.Clear()
    v4_entity.property_list().append(self.__v4_string_property(PROPERTY_NAME_EMAIL, v3_user_value.email(), False))
    v4_entity.property_list().append(self.__v4_string_property(PROPERTY_NAME_AUTH_DOMAIN, v3_user_value.auth_domain(), False))
    if v3_user_value.gaiaid() != 0:
        v4_entity.property_list().append(self.__v4_integer_property(PROPERTY_NAME_INTERNAL_ID, v3_user_value.gaiaid(), False))
    if v3_user_value.has_obfuscated_gaiaid():
        v4_entity.property_list().append(self.__v4_string_property(PROPERTY_NAME_USER_ID, v3_user_value.obfuscated_gaiaid(), False))
    if v3_user_value.has_federated_identity():
        v4_entity.property_list().append(self.__v4_string_property(PROPERTY_NAME_FEDERATED_IDENTITY, v3_user_value.federated_identity(), False))
    if v3_user_value.has_federated_provider():
        v4_entity.property_list().append(self.__v4_string_property(PROPERTY_NAME_FEDERATED_PROVIDER, v3_user_value.federated_provider(), False))