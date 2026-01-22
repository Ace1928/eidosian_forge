from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def __v3_reference_value_to_v3_reference(self, v3_ref_value, v3_ref):
    """Converts a v3 ReferenceValue to a v3 Reference.

    Args:
      v3_ref_value: an entity_pb.PropertyValue_ReferenceValue
      v3_ref: an entity_pb.Reference to populate
    """
    v3_ref.Clear()
    if v3_ref_value.has_app():
        v3_ref.set_app(v3_ref_value.app())
    if v3_ref_value.has_name_space():
        v3_ref.set_name_space(v3_ref_value.name_space())
    for v3_ref_value_path_element in v3_ref_value.pathelement_list():
        v3_path_element = v3_ref.mutable_path().add_element()
        if v3_ref_value_path_element.has_type():
            v3_path_element.set_type(v3_ref_value_path_element.type())
        if v3_ref_value_path_element.has_id():
            v3_path_element.set_id(v3_ref_value_path_element.id())
        if v3_ref_value_path_element.has_name():
            v3_path_element.set_name(v3_ref_value_path_element.name())