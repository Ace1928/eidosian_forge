from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
def _get_paths(base_dir, parser):
    """Gets a list of Paths in a given directory.

  Args:
    base_dir: directory.
    parser: a function which gets the raw Path and can augment it with
      information such as the export_version, or ignore the path by returning
      None.  An example parser may extract the export version from a path such
      as "/tmp/exports/100" an another may extract from a full file name such as
      "/tmp/checkpoint-99.out".

  Returns:
    A list of Paths contained in the base directory with the parsing function
    applied.
    By default the following fields are populated,
      - Path.path
    The parsing function is responsible for populating,
      - Path.export_version
  """
    raw_paths = gfile.ListDirectory(base_dir)
    paths = []
    for r in raw_paths:
        r = tf.compat.as_str_any(r)
        if r[-1] == '/':
            r = r[0:len(r) - 1]
        p = parser(Path(os.path.join(tf.compat.as_str_any(base_dir), r), None))
        if p:
            paths.append(p)
    return sorted(paths)