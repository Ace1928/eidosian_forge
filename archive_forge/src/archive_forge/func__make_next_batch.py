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
def _make_next_batch(self, fetch_options):
    in_memory_offset = FetchOptions.offset(fetch_options)
    augmented_query = self._batch_shared.augmented_query
    if in_memory_offset and (augmented_query._in_memory_filter or augmented_query._in_memory_results):
        fetch_options = FetchOptions(offset=0)
    else:
        in_memory_offset = None
    return (fetch_options, _AugmentedBatch(self._batch_shared, in_memory_offset=in_memory_offset, in_memory_limit=self.__in_memory_limit, start_cursor=self.end_cursor, next_index=self.__next_index))