import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
def _dispatch_command(self, command):
    """Dispatch user command.

    Args:
      command: (str) Command to dispatch.

    Returns:
      An exit token object. None value means that the UI loop should not exit.
      A non-None value means the UI loop should exit.
    """
    if command in self.CLI_EXIT_COMMANDS:
        return debugger_cli_common.EXPLICIT_USER_EXIT
    try:
        prefix, args, output_file_path = self._parse_command(command)
    except SyntaxError as e:
        print(str(e))
        return
    if self._command_handler_registry.is_registered(prefix):
        try:
            screen_output = self._command_handler_registry.dispatch_command(prefix, args, screen_info=None)
        except debugger_cli_common.CommandLineExit as e:
            return e.exit_token
    else:
        screen_output = debugger_cli_common.RichTextLines([self.ERROR_MESSAGE_PREFIX + 'Invalid command prefix "%s"' % prefix])
    self._display_output(screen_output)
    if output_file_path:
        try:
            screen_output.write_to_file(output_file_path)
            print('Wrote output to %s' % output_file_path)
        except Exception:
            print('Failed to write output to %s' % output_file_path)