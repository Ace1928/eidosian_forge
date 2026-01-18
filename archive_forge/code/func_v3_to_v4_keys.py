from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_to_v4_keys(self, v3_refs):
    """Converts a list of v3 References to a list of v4 Keys.

    Args:
      v3_refs: a list of entity_pb.Reference objects

    Returns:
      a list of entity_v4_pb.Key objects
    """
    v4_keys = []
    for v3_ref in v3_refs:
        v4_key = entity_v4_pb.Key()
        self.v3_to_v4_key(v3_ref, v4_key)
        v4_keys.append(v4_key)
    return v4_keys