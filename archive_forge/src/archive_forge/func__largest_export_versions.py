from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
def _largest_export_versions(n):
    """Creates a filter that keeps the largest n export versions.

  Args:
    n: number of versions to keep.

  Returns:
    A filter function that keeps the n largest paths.
  """

    def keep(paths):
        heap = []
        for idx, path in enumerate(paths):
            if path.export_version is not None:
                heapq.heappush(heap, (path.export_version, idx))
        keepers = [paths[i] for _, i in heapq.nlargest(n, heap)]
        return sorted(keepers)
    return keep