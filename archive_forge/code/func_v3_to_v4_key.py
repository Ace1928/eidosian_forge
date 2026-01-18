from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_to_v4_key(self, v3_ref, v4_key):
    """Converts a v3 Reference to a v4 Key.

    Args:
      v3_ref: an entity_pb.Reference
      v4_key: an entity_v4_pb.Key to populate
    """
    v4_key.Clear()
    if not v3_ref.app():
        return
    v4_key.mutable_partition_id().set_dataset_id(v3_ref.app())
    if v3_ref.name_space():
        v4_key.mutable_partition_id().set_namespace(v3_ref.name_space())
    for v3_element in v3_ref.path().element_list():
        v4_element = v4_key.add_path_element()
        v4_element.set_kind(v3_element.type())
        if v3_element.has_id():
            v4_element.set_id(v3_element.id())
        if v3_element.has_name():
            v4_element.set_name(v3_element.name())