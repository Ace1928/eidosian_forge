from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __is_v3_property_value_meaning_valid(self, v3_property_value, v3_meaning):
    """Returns True if the v3 PropertyValue's type value matches its meaning."""

    def ReturnTrue():
        return True

    def HasStringValue():
        return v3_property_value.has_stringvalue()

    def HasInt64Value():
        return v3_property_value.has_int64value()

    def HasPointValue():
        return v3_property_value.has_pointvalue()

    def ReturnFalse():
        return False
    value_checkers = {entity_pb.Property.NO_MEANING: ReturnTrue, entity_pb.Property.INDEX_VALUE: ReturnTrue, entity_pb.Property.BLOB: HasStringValue, entity_pb.Property.TEXT: HasStringValue, entity_pb.Property.BYTESTRING: HasStringValue, entity_pb.Property.ATOM_CATEGORY: HasStringValue, entity_pb.Property.ATOM_LINK: HasStringValue, entity_pb.Property.ATOM_TITLE: HasStringValue, entity_pb.Property.ATOM_CONTENT: HasStringValue, entity_pb.Property.ATOM_SUMMARY: HasStringValue, entity_pb.Property.ATOM_AUTHOR: HasStringValue, entity_pb.Property.GD_EMAIL: HasStringValue, entity_pb.Property.GD_IM: HasStringValue, entity_pb.Property.GD_PHONENUMBER: HasStringValue, entity_pb.Property.GD_POSTALADDRESS: HasStringValue, entity_pb.Property.BLOBKEY: HasStringValue, entity_pb.Property.ENTITY_PROTO: HasStringValue, entity_pb.Property.GD_WHEN: HasInt64Value, entity_pb.Property.GD_RATING: HasInt64Value, entity_pb.Property.GEORSS_POINT: HasPointValue, entity_pb.Property.EMPTY_LIST: ReturnTrue}
    default = ReturnFalse
    return value_checkers.get(v3_meaning, default)()