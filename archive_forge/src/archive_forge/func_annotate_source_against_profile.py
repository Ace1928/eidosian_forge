import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def annotate_source_against_profile(profile_data, source_file_path, node_name_filter=None, op_type_filter=None, min_line=None, max_line=None):
    """Annotate a Python source file with profiling information at each line.

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
  """
    source_file_path = _norm_abs_path(source_file_path)
    node_name_regex = re.compile(node_name_filter) if node_name_filter else None
    op_type_regex = re.compile(op_type_filter) if op_type_filter else None
    line_to_profile_summary = {}
    for profile_datum in profile_data:
        if not profile_datum.file_path:
            continue
        if _norm_abs_path(profile_datum.file_path) != source_file_path:
            continue
        if min_line is not None and profile_datum.line_number < min_line or (max_line is not None and profile_datum.line_number >= max_line):
            continue
        if node_name_regex and (not node_name_regex.match(profile_datum.node_exec_stats.node_name)):
            continue
        if op_type_regex and (not op_type_regex.match(profile_datum.op_type)):
            continue
        if profile_datum.line_number not in line_to_profile_summary:
            line_to_profile_summary[profile_datum.line_number] = profiling.AggregateProfile(profile_datum)
        else:
            line_to_profile_summary[profile_datum.line_number].add(profile_datum)
    return line_to_profile_summary