import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _dump_file_name_to_datum(self, dir_name, file_name):
    """Obtain a DebugTensorDatum from the directory and file name.

    Args:
      dir_name: (`str`) Name of the directory in which the dump file resides.
      file_name: (`str`) Base name of the dump file.

    Returns:
      (`DebugTensorDatum`) The `DebugTensorDatum` loaded from the dump file.
    """
    debug_dump_rel_path = os.path.join(os.path.relpath(dir_name, self._dump_root), file_name)
    return DebugTensorDatum(self._dump_root, debug_dump_rel_path)