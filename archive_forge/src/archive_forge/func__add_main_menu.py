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
def _add_main_menu(output, node_name=None, enable_list_tensors=True, enable_node_info=True, enable_print_tensor=True, enable_list_inputs=True, enable_list_outputs=True):
    """Generate main menu for the screen output from a command.

  Args:
    output: (debugger_cli_common.RichTextLines) the output object to modify.
    node_name: (str or None) name of the node involved (if any). If None,
      the menu items node_info, list_inputs and list_outputs will be
      automatically disabled, overriding the values of arguments
      enable_node_info, enable_list_inputs and enable_list_outputs.
    enable_list_tensors: (bool) whether the list_tensor menu item will be
      enabled.
    enable_node_info: (bool) whether the node_info item will be enabled.
    enable_print_tensor: (bool) whether the print_tensor item will be enabled.
    enable_list_inputs: (bool) whether the item list_inputs will be enabled.
    enable_list_outputs: (bool) whether the item list_outputs will be enabled.
  """
    menu = debugger_cli_common.Menu()
    menu.append(debugger_cli_common.MenuItem('list_tensors', 'list_tensors', enabled=enable_list_tensors))
    if node_name:
        menu.append(debugger_cli_common.MenuItem('node_info', 'node_info -a -d -t %s' % node_name, enabled=enable_node_info))
        menu.append(debugger_cli_common.MenuItem('print_tensor', 'print_tensor %s' % node_name, enabled=enable_print_tensor))
        menu.append(debugger_cli_common.MenuItem('list_inputs', 'list_inputs -c -r %s' % node_name, enabled=enable_list_inputs))
        menu.append(debugger_cli_common.MenuItem('list_outputs', 'list_outputs -c -r %s' % node_name, enabled=enable_list_outputs))
    else:
        menu.append(debugger_cli_common.MenuItem('node_info', None, enabled=False))
        menu.append(debugger_cli_common.MenuItem('print_tensor', None, enabled=False))
        menu.append(debugger_cli_common.MenuItem('list_inputs', None, enabled=False))
        menu.append(debugger_cli_common.MenuItem('list_outputs', None, enabled=False))
    menu.append(debugger_cli_common.MenuItem('run_info', 'run_info'))
    menu.append(debugger_cli_common.MenuItem('help', 'help'))
    output.annotations[debugger_cli_common.MAIN_MENU_KEY] = menu