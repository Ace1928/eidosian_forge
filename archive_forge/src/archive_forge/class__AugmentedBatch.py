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
class _AugmentedBatch(Batch):
    """A batch produced by a datastore_query._AugmentedQuery."""

    @classmethod
    @datastore_rpc._positional(5)
    def create_async(cls, augmented_query, query_options, conn, req, in_memory_offset, in_memory_limit, start_cursor):
        initial_offset = 0 if in_memory_offset is not None else None
        batch_shared = _BatchShared(augmented_query._query, query_options, conn, augmented_query, initial_offset=initial_offset)
        batch0 = cls(batch_shared, in_memory_offset=in_memory_offset, in_memory_limit=in_memory_limit, start_cursor=start_cursor)
        return batch0._make_query_rpc_call(query_options, req)

    @datastore_rpc._positional(2)
    def __init__(self, batch_shared, in_memory_offset=None, in_memory_limit=None, next_index=0, start_cursor=Cursor()):
        """A Constructor for datastore_query._AugmentedBatch.

    Constructed by datastore_query._AugmentedQuery. Should not be called
    directly.
    """
        super(_AugmentedBatch, self).__init__(batch_shared, start_cursor=start_cursor)
        self.__in_memory_offset = in_memory_offset
        self.__in_memory_limit = in_memory_limit
        self.__next_index = next_index

    @property
    def query(self):
        """The query the current batch came from."""
        return self._batch_shared.augmented_query

    def cursor(self, index):
        raise NotImplementedError

    def _extend(self, next_batch):
        super(_AugmentedBatch, self)._extend(next_batch)
        self.__in_memory_limit = next_batch.__in_memory_limit
        self.__in_memory_offset = next_batch.__in_memory_offset
        self.__next_index = next_batch.__next_index

    def _process_v1_results(self, results):
        """Process V4 results by converting to V3 and calling _process_results."""
        v3_results = []
        is_projection = bool(self.query_options.projection)
        for v1_result in results:
            v3_entity = entity_pb.EntityProto()
            self._batch_shared.conn.adapter.get_entity_converter().v1_to_v3_entity(v1_result.entity, v3_entity, is_projection)
            v3_results.append(v3_entity)
        return self._process_results(v3_results)

    def _process_results(self, results):
        in_memory_filter = self._batch_shared.augmented_query._in_memory_filter
        if in_memory_filter:
            results = list(filter(in_memory_filter, results))
        in_memory_results = self._batch_shared.augmented_query._in_memory_results
        if in_memory_results and self.__next_index < len(in_memory_results):
            original_query = super(_AugmentedBatch, self).query
            if original_query._order:
                if results:
                    next_result = in_memory_results[self.__next_index]
                    next_key = original_query._order.key(next_result)
                    i = 0
                    while i < len(results):
                        result = results[i]
                        result_key = original_query._order.key(result)
                        while next_key <= result_key:
                            results.insert(i, next_result)
                            i += 1
                            self.__next_index += 1
                            if self.__next_index >= len(in_memory_results):
                                break
                            next_result = in_memory_results[self.__next_index]
                            next_key = original_query._order.key(next_result)
                        i += 1
            elif results or not super(_AugmentedBatch, self).more_results:
                results = in_memory_results + results
                self.__next_index = len(in_memory_results)
        if self.__in_memory_offset:
            assert not self._skipped_results
            offset = min(self.__in_memory_offset, len(results))
            if offset:
                self._skipped_results += offset
                self.__in_memory_offset -= offset
                results = results[offset:]
        if self.__in_memory_limit is not None:
            results = results[:self.__in_memory_limit]
            self.__in_memory_limit -= len(results)
            if self.__in_memory_limit <= 0:
                self._end()
        return super(_AugmentedBatch, self)._process_results(results)

    def _make_next_batch(self, fetch_options):
        in_memory_offset = FetchOptions.offset(fetch_options)
        augmented_query = self._batch_shared.augmented_query
        if in_memory_offset and (augmented_query._in_memory_filter or augmented_query._in_memory_results):
            fetch_options = FetchOptions(offset=0)
        else:
            in_memory_offset = None
        return (fetch_options, _AugmentedBatch(self._batch_shared, in_memory_offset=in_memory_offset, in_memory_limit=self.__in_memory_limit, start_cursor=self.end_cursor, next_index=self.__next_index))