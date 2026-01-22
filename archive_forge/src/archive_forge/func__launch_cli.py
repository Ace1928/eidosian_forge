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
def _launch_cli(self):
    """Launch the interactive command-line interface.

    Returns:
      The OnRunStartResponse specified by the user using the "run" command.
    """
    self._register_this_run_info(self._run_cli)
    response = self._run_cli.run_ui(init_command=self._init_command, title=self._title, title_color=self._title_color)
    return response