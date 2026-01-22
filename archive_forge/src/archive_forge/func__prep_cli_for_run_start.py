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
def _prep_cli_for_run_start(self):
    """Prepare (but not launch) the CLI for run-start."""
    self._run_cli = ui_factory.get_ui(self._ui_type, config=self._config)
    help_intro = debugger_cli_common.RichTextLines([])
    if self._run_call_count == 1:
        help_intro.extend(cli_shared.get_tfdbg_logo())
        help_intro.extend(debugger_cli_common.get_tensorflow_version_lines())
    help_intro.extend(debugger_cli_common.RichTextLines('Upcoming run:'))
    help_intro.extend(self._run_info)
    self._run_cli.set_help_intro(help_intro)
    self._title = 'run-start: ' + self._run_description
    self._init_command = 'run_info'
    self._title_color = 'blue_on_white'