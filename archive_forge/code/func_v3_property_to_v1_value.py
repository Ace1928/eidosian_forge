from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_property_to_v1_value(self, v3_property, indexed, v1_value):
    """Converts a v3 Property to a v1 Value.

    Args:
      v3_property: an entity_pb.Property
      indexed: whether the v3 property is indexed
      v1_value: an googledatastore.Value to populate
    """
    v1_value.Clear()
    v3_property_value = v3_property.value()
    v3_meaning = v3_property.meaning()
    v3_uri_meaning = None
    if v3_property.meaning_uri():
        v3_uri_meaning = v3_property.meaning_uri()
    if not self.__is_v3_property_value_union_valid(v3_property_value):
        v3_meaning = None
        v3_uri_meaning = None
    elif v3_meaning == entity_pb.Property.NO_MEANING:
        v3_meaning = None
    elif not self.__is_v3_property_value_meaning_valid(v3_property_value, v3_meaning):
        v3_meaning = None
    is_zlib_value = False
    if v3_uri_meaning:
        if v3_uri_meaning == URI_MEANING_ZLIB:
            if v3_property_value.has_stringvalue():
                is_zlib_value = True
                if v3_meaning != entity_pb.Property.BLOB:
                    v3_meaning = entity_pb.Property.BLOB
            else:
                pass
        else:
            pass
    if v3_property.meaning() == entity_pb.Property.EMPTY_LIST:
        v1_value.array_value.values.extend([])
        v3_meaning = None
    elif v3_property_value.has_booleanvalue():
        v1_value.boolean_value = v3_property_value.booleanvalue()
    elif v3_property_value.has_int64value():
        if v3_meaning == entity_pb.Property.GD_WHEN and is_in_rfc_3339_bounds(v3_property_value.int64value()):
            googledatastore.helper.micros_to_timestamp(v3_property_value.int64value(), v1_value.timestamp_value)
            v3_meaning = None
        else:
            v1_value.integer_value = v3_property_value.int64value()
    elif v3_property_value.has_doublevalue():
        v1_value.double_value = v3_property_value.doublevalue()
    elif v3_property_value.has_referencevalue():
        v3_ref = entity_pb.Reference()
        self.__v3_reference_value_to_v3_reference(v3_property_value.referencevalue(), v3_ref)
        self.v3_to_v1_key(v3_ref, v1_value.key_value)
    elif v3_property_value.has_stringvalue():
        if v3_meaning == entity_pb.Property.ENTITY_PROTO:
            serialized_entity_v3 = v3_property_value.stringvalue()
            v3_entity = entity_pb.EntityProto()
            v3_entity.ParsePartialFromString(serialized_entity_v3)
            self.v3_to_v1_entity(v3_entity, v1_value.entity_value)
            v3_meaning = None
        elif v3_meaning == entity_pb.Property.BLOB or v3_meaning == entity_pb.Property.BYTESTRING:
            v1_value.blob_value = v3_property_value.stringvalue()
            if indexed or v3_meaning == entity_pb.Property.BLOB:
                v3_meaning = None
        else:
            string_value = v3_property_value.stringvalue()
            if is_valid_utf8(string_value):
                v1_value.string_value = string_value
            else:
                v1_value.blob_value = string_value
                if v3_meaning != entity_pb.Property.INDEX_VALUE:
                    v3_meaning = None
    elif v3_property_value.has_pointvalue():
        if v3_meaning != MEANING_GEORSS_POINT:
            v1_value.meaning = MEANING_POINT_WITHOUT_V3_MEANING
        point_value = v3_property_value.pointvalue()
        v1_value.geo_point_value.latitude = point_value.x()
        v1_value.geo_point_value.longitude = point_value.y()
        v3_meaning = None
    elif v3_property_value.has_uservalue():
        self.v3_user_value_to_v1_entity(v3_property_value.uservalue(), v1_value.entity_value)
        v1_value.meaning = MEANING_PREDEFINED_ENTITY_USER
        v3_meaning = None
    else:
        v1_value.null_value = googledatastore.NULL_VALUE
    if is_zlib_value:
        v1_value.meaning = MEANING_ZLIB
    elif v3_meaning:
        v1_value.meaning = v3_meaning
    if indexed == v1_value.exclude_from_indexes:
        v1_value.exclude_from_indexes = not indexed