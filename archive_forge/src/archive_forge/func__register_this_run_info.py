import argparse
import os
import sys
import tempfile
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.lib.io import file_io
def _register_this_run_info(self, curses_cli):
    curses_cli.register_command_handler('run', self._run_handler, self._argparsers['run'].format_help(), prefix_aliases=['r'])
    curses_cli.register_command_handler('run_info', self._run_info_handler, self._argparsers['run_info'].format_help(), prefix_aliases=['ri'])
    curses_cli.register_command_handler('print_feed', self._print_feed_handler, self._argparsers['print_feed'].format_help(), prefix_aliases=['pf'])
    if self._tensor_filters:
        curses_cli.register_tab_comp_context(['run', 'r'], list(self._tensor_filters.keys()))
    if self._feed_dict and hasattr(self._feed_dict, 'keys'):
        feed_keys = [common.get_graph_element_name(key) for key in self._feed_dict.keys()]
        curses_cli.register_tab_comp_context(['print_feed', 'pf'], feed_keys)