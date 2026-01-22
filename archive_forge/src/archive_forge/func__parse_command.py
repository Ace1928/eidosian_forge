import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def _parse_command(self, command):
    """Parse a command string into prefix and arguments.

    Args:
      command: (str) Command string to be parsed.

    Returns:
      prefix: (str) The command prefix.
      args: (list of str) The command arguments (i.e., not including the
        prefix).
      output_file_path: (str or None) The path to save the screen output
        to (if any).
    """
    command = command.strip()
    if not command:
        return ('', [], None)
    command_items = command_parser.parse_command(command)
    command_items, output_file_path = command_parser.extract_output_file_path(command_items)
    return (command_items[0], command_items[1:], output_file_path)