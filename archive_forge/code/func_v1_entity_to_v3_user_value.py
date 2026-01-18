from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v1_entity_to_v3_user_value(self, v1_user_entity, v3_user_value):
    """Converts a v1 user Entity to a v3 UserValue.

    Args:
      v1_user_entity: an googledatastore.Entity representing a user
      v3_user_value: an entity_pb.Property_UserValue to populate
    """
    v3_user_value.Clear()
    properties = v1_user_entity.properties
    v3_user_value.set_email(self.__get_v1_string_value(properties[PROPERTY_NAME_EMAIL]))
    v3_user_value.set_auth_domain(self.__get_v1_string_value(properties[PROPERTY_NAME_AUTH_DOMAIN]))
    if PROPERTY_NAME_USER_ID in properties:
        v3_user_value.set_obfuscated_gaiaid(self.__get_v1_string_value(properties[PROPERTY_NAME_USER_ID]))
    if PROPERTY_NAME_INTERNAL_ID in properties:
        v3_user_value.set_gaiaid(self.__get_v1_integer_value(properties[PROPERTY_NAME_INTERNAL_ID]))
    else:
        v3_user_value.set_gaiaid(0)
    if PROPERTY_NAME_FEDERATED_IDENTITY in properties:
        v3_user_value.set_federated_identity(self.__get_v1_string_value(properties[PROPERTY_NAME_FEDERATED_IDENTITY]))
    if PROPERTY_NAME_FEDERATED_PROVIDER in properties:
        v3_user_value.set_federated_provider(self.__get_v1_string_value(properties[PROPERTY_NAME_FEDERATED_PROVIDER]))