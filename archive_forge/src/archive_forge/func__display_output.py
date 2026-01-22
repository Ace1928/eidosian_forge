import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
def _display_output(self, screen_output):
    for line in screen_output.lines:
        print(line)