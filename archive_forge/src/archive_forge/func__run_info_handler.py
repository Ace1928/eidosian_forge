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
def _run_info_handler(self, args, screen_info=None):
    output = debugger_cli_common.RichTextLines([])
    if self._run_call_count == 1:
        output.extend(cli_shared.get_tfdbg_logo())
        output.extend(debugger_cli_common.get_tensorflow_version_lines())
    output.extend(self._run_info)
    if not self._is_run_start and debugger_cli_common.MAIN_MENU_KEY in output.annotations:
        menu = output.annotations[debugger_cli_common.MAIN_MENU_KEY]
        if 'list_tensors' not in menu.captions():
            menu.insert(0, debugger_cli_common.MenuItem('list_tensors', 'list_tensors'))
    return output