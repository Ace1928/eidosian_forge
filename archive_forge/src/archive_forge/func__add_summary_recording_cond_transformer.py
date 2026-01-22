import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _add_summary_recording_cond_transformer(parent, node, full_name, name, logs, cond):
    """Adds cond argument to tf.contrib.summary.xxx_record_summaries().

  This is in anticipation of them being renamed to tf.summary.record_if(), which
  requires the cond argument.
  """
    node.args.append(pasta.parse(cond))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Adding `%s` argument to %s in anticipation of it being renamed to tf.compat.v2.summary.record_if()' % (cond, full_name or name)))
    return node