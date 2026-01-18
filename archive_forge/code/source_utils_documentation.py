import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
Annotate a Python source file with profiling information at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    profile_data: (`list` of `ProfileDatum`) A list of `ProfileDatum`.
    source_file_path: (`str`) Path to the source file being annotated.
    node_name_filter: Regular expression to filter by node name.
    op_type_filter: Regular expression to filter by op type.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a the namedtuple
      `profiling.LineOrFuncProfileSummary`.
  