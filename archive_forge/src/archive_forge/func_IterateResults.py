import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
def IterateResults(results):
    """Iterator function to return all results in sorted order.

      Iterate over the array of results, yielding the next element, in
      sorted order. This function is destructive (results will be empty
      when the operation is complete).

      Args:
        results: list of result iterators to merge and iterate through

      Yields:
        The next result in sorted order.
      """
    result_heap = []
    for result in results:
        heap_value = MultiQuery.SortOrderEntity(result, self.__orderings)
        if heap_value.GetEntity():
            heapq.heappush(result_heap, heap_value)
    used_keys = set()
    while result_heap:
        if upper_bound is not None and len(used_keys) >= upper_bound:
            break
        top_result = heapq.heappop(result_heap)
        dedupe_key = GetDedupeKey(top_result)
        if dedupe_key not in used_keys:
            result = top_result.GetEntity()
            if override:
                for key in result.keys():
                    if key not in projection:
                        del result[key]
            yield result
        else:
            pass
        used_keys.add(dedupe_key)
        results_to_push = []
        while result_heap:
            next = heapq.heappop(result_heap)
            if dedupe_key != GetDedupeKey(next):
                results_to_push.append(next)
                break
            else:
                results_to_push.append(next.GetNext())
        results_to_push.append(top_result.GetNext())
        for popped_result in results_to_push:
            if popped_result.GetEntity():
                heapq.heappush(result_heap, popped_result)