import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
def _ui_loop(self):
    while True:
        command = self._get_user_command()
        exit_token = self._dispatch_command(command)
        if exit_token is not None:
            return exit_token