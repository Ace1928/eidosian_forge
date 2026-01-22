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
def _sort_dump_data_by(self, data, sort_by, reverse):
    """Sort a list of DebugTensorDatum in specified order.

    Args:
      data: (list of DebugTensorDatum) the data to be sorted.
      sort_by: The field to sort data by.
      reverse: (bool) Whether to use reversed (descending) order.

    Returns:
      (list of DebugTensorDatum) in sorted order.

    Raises:
      ValueError: given an invalid value of sort_by.
    """
    if sort_by == SORT_TENSORS_BY_TIMESTAMP:
        return sorted(data, reverse=reverse, key=lambda x: x.timestamp)
    elif sort_by == SORT_TENSORS_BY_DUMP_SIZE:
        return sorted(data, reverse=reverse, key=lambda x: x.dump_size_bytes)
    elif sort_by == SORT_TENSORS_BY_OP_TYPE:
        return sorted(data, reverse=reverse, key=lambda x: self._debug_dump.node_op_type(x.node_name))
    elif sort_by == SORT_TENSORS_BY_TENSOR_NAME:
        return sorted(data, reverse=reverse, key=lambda x: '%s:%d' % (x.node_name, x.output_slot))
    else:
        raise ValueError('Unsupported key to sort tensors by: %s' % sort_by)