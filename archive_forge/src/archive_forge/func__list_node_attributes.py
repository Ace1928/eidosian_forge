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
def _list_node_attributes(self, node_name):
    """List neighbors (inputs or recipients) of a node.

    Args:
      node_name: Name of the node of which the attributes are to be listed.

    Returns:
      A RichTextLines object.
    """
    lines = []
    lines.append('')
    lines.append('Node attributes:')
    attrs = self._debug_dump.node_attributes(node_name)
    for attr_key in attrs:
        lines.append('  %s:' % attr_key)
        attr_val_str = repr(attrs[attr_key]).strip().replace('\n', ' ')
        lines.append('    %s' % attr_val_str)
        lines.append('')
    return debugger_cli_common.RichTextLines(lines)