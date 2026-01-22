from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
def _process_v1_results(self, results):
    """Process V4 results by converting to V3 and calling _process_results."""
    v3_results = []
    is_projection = bool(self.query_options.projection)
    for v1_result in results:
        v3_entity = entity_pb.EntityProto()
        self._batch_shared.conn.adapter.get_entity_converter().v1_to_v3_entity(v1_result.entity, v3_entity, is_projection)
        v3_results.append(v3_entity)
    return self._process_results(v3_results)