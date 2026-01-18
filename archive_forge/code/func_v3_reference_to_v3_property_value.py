from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_reference_to_v3_property_value(self, v3_ref, v3_property_value):
    """Converts a v3 Reference to a v3 PropertyValue.

    Args:
      v3_ref: an entity_pb.Reference
      v3_property_value: an entity_pb.PropertyValue to populate
    """
    v3_property_value.Clear()
    reference_value = v3_property_value.mutable_referencevalue()
    if v3_ref.has_app():
        reference_value.set_app(v3_ref.app())
    if v3_ref.has_name_space():
        reference_value.set_name_space(v3_ref.name_space())
    for v3_path_element in v3_ref.path().element_list():
        v3_ref_value_path_element = reference_value.add_pathelement()
        if v3_path_element.has_type():
            v3_ref_value_path_element.set_type(v3_path_element.type())
        if v3_path_element.has_id():
            v3_ref_value_path_element.set_id(v3_path_element.id())
        if v3_path_element.has_name():
            v3_ref_value_path_element.set_name(v3_path_element.name())