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
def _prep_debug_cli_for_run_end(self, debug_dump, tf_error, passed_filter, passed_filter_exclude_node_names):
    """Prepare (but not launch) CLI for run-end, with debug dump from the run.

    Args:
      debug_dump: (debug_data.DebugDumpDir) The debug dump directory from this
        run.
      tf_error: (None or OpError) OpError that happened during the run() call
        (if any).
      passed_filter: (None or str) Name of the tensor filter that just passed
        and caused the preparation of this run-end CLI (if any).
      passed_filter_exclude_node_names: (None or str) Regular expression used
        with the tensor filter to exclude ops with names matching the regular
        expression.
    """
    if tf_error:
        help_intro = cli_shared.get_error_intro(tf_error)
        self._init_command = 'help'
        self._title_color = 'red_on_white'
    else:
        help_intro = None
        self._init_command = 'lt'
        self._title_color = 'black_on_white'
        if passed_filter is not None:
            self._init_command = 'lt -f %s' % passed_filter
            if passed_filter_exclude_node_names:
                self._init_command += ' --filter_exclude_node_names %s' % passed_filter_exclude_node_names
            self._title_color = 'red_on_white'
    self._run_cli = analyzer_cli.create_analyzer_ui(debug_dump, self._tensor_filters, ui_type=self._ui_type, on_ui_exit=self._remove_dump_root, config=self._config)
    dumped_tensor_names = []
    for datum in debug_dump.dumped_tensor_data:
        dumped_tensor_names.append('%s:%d' % (datum.node_name, datum.output_slot))
    self._run_cli.register_tab_comp_context(['print_tensor', 'pt'], dumped_tensor_names)
    self._run_cli.register_tab_comp_context(['node_info', 'ni', 'list_inputs', 'li', 'list_outputs', 'lo'], [str(node_name) for node_name in debug_dump.nodes()])
    self._title = 'run-end: ' + self._run_description
    if help_intro:
        self._run_cli.set_help_intro(help_intro)