from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_reference_to_group(self, v3_ref, group):
    """Converts a v3 Reference to a v3 Path representing the entity group.

    The entity group is represented as an entity_pb.Path containing only the
    first element in the provided Reference.

    Args:
      v3_ref: an entity_pb.Reference
      group: an entity_pb.Path to populate
    """
    group.Clear()
    path = v3_ref.path()
    assert path.element_size() >= 1
    group.add_element().CopyFrom(path.element(0))