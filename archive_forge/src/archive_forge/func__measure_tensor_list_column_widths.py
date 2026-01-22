import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def _measure_tensor_list_column_widths(self, data):
    """Determine the maximum widths of the timestamp and op-type column.

    This method assumes that data is sorted in the default order, i.e.,
    by ascending timestamps.

    Args:
      data: (list of DebugTensorDaum) the data based on which the maximum
        column widths will be determined.

    Returns:
      (int) maximum width of the timestamp column. 0 if data is empty.
      (int) maximum width of the dump size column. 0 if data is empty.
      (int) maximum width of the op type column. 0 if data is empty.
    """
    max_timestamp_width = 0
    if data:
        max_rel_time_ms = (data[-1].timestamp - self._debug_dump.t0) / 1000.0
        max_timestamp_width = len('[%.3f] ' % max_rel_time_ms) + 1
    max_timestamp_width = max(max_timestamp_width, len(self._TIMESTAMP_COLUMN_HEAD) + 1)
    max_dump_size_width = 0
    for dump in data:
        dump_size_str = cli_shared.bytes_to_readable_str(dump.dump_size_bytes)
        if len(dump_size_str) + 1 > max_dump_size_width:
            max_dump_size_width = len(dump_size_str) + 1
    max_dump_size_width = max(max_dump_size_width, len(self._DUMP_SIZE_COLUMN_HEAD) + 1)
    max_op_type_width = 0
    for dump in data:
        op_type = self._debug_dump.node_op_type(dump.node_name)
        if len(op_type) + 1 > max_op_type_width:
            max_op_type_width = len(op_type) + 1
    max_op_type_width = max(max_op_type_width, len(self._OP_TYPE_COLUMN_HEAD) + 1)
    return (max_timestamp_width, max_dump_size_width, max_op_type_width)